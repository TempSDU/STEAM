import argparse
import os
import time
import numpy as np
import torch
import torch.utils.data as Data
import torch.optim as optim
from torch import nn
from model import model
from dataLoader import TrainDataset, ValidDataset, TestDataset
from script import init_seeds, xavier_init, seqs_normalization, seqs_normalization_test, evaluate_function, get_metrics, \
    cal_similarity, stat_operation

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-ef', type=str, default='./Beauty/test.dat', help='path of test dataset')
    parser.add_argument('-en', type=str, default='./Beauty/test_neg.dat', help='path of test dataset(negative instance)')
    parser.add_argument('-b', type=int, default=256, help='batch_size')
    parser.add_argument('-dr', type=float, default=0.5, help='normalization')
    parser.add_argument('-hd', type=int, default=64, help='hidden_layer_dimension')
    parser.add_argument('-hn', type=int, default=1, help='hidden_layer_dimension')
    parser.add_argument('-ln', type=int, default=1, help='number_of_transformer_layer')
    parser.add_argument('-o', type=str, default='./save_model/', help='save_path')
    parser.add_argument('-n', type=int, default=12104, help='item_num')  # Beauty dataset contains 12101 items, plus padding, EOS, mask
    parser.add_argument('-ml', type=int, default=50, help='max_seqs_len')
    parser.add_argument('-mml', type=int, default=60, help='modified_max_seqs_len')
    parser.add_argument('-mi', type=int, default=5, help='max_insert_num')
    parser.add_argument('-e', type=int, help='epoch to be evaluate')
    args = parser.parse_args()
    test_file = args.ef
    test_neg_file = args.en
    batch_size = args.b
    dropout_rate = args.dr
    hidden_unit = args.hd
    head_num = args.hn
    layer_num = args.ln
    save_path = args.o
    item_num = args.n
    max_seqs_len = args.ml
    modified_max_seqs_len = args.mml
    max_insert_num = args.mi
    epoch = args.e

    init_seeds()

    dataset = TestDataset(test_file, test_neg_file, item_num, modified_max_seqs_len)
    dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    model = model(dropout_rate, hidden_unit, item_num, head_num, layer_num, max_insert_num)
    model.load_state_dict(torch.load(save_path + 'model/steam-' + str(epoch) + '.pth'))
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    total_result = []  # For the modified part of the dataset, the metrics before modification
    total_result_modified = []  # For the modified part of the dataset, the metrics after modification
    total_result2 = []  # For the unmodified part of the dataset, the metrics before modification
    total_result_modified2 = []  # For the unmodified part of the dataset, the metrics after modification (equal to the upper one)
    seq_list = []
    seq_mod_list = []
    indices = []

    with torch.no_grad():
        index = 0
        for batch in dataloader:
            _, test_neg, masked_seq, leave_one_seq, target = batch
            if torch.cuda.is_available():
                test_neg = test_neg.cuda()
                masked_seq = masked_seq.cuda()
                leave_one_seq = leave_one_seq.cuda()
                target = target.cuda()
            # original sequence
            recommender_output = model.forward(masked_seq)
            recommender_output = recommender_output[masked_seq == item_num - 1]
            result = evaluate_function(recommender_output, target, test_neg)
            # corrected sequence
            full_layer_output_ori, insert_net_output_ori = model.corrector_inference(leave_one_seq)
            modified_leave_one_seqs = model.seqs_correction(full_layer_output_ori, insert_net_output_ori,
                                                            leave_one_seq)
            modified_masked_seqs = seqs_normalization_test(modified_leave_one_seqs, modified_max_seqs_len, item_num)
            if torch.cuda.is_available():
                modified_masked_seqs = modified_masked_seqs.cuda()
            recommender_modified_output = model.forward(modified_masked_seqs)
            recommender_modified_output = recommender_modified_output[modified_masked_seqs == item_num - 1]
            result_modified = evaluate_function(recommender_modified_output, target, test_neg)
            similarity = cal_similarity(masked_seq, modified_masked_seqs)
            for sim, res, res_mod, seq, seq_mod in zip(similarity, result, result_modified, masked_seq,
                                                       modified_masked_seqs):
                if sim == 0:
                    total_result.append(res)
                    total_result_modified.append(res_mod)
                    seq_list.append(seq.tolist())
                    seq_mod_list.append(seq_mod.tolist())
                    indices.append(index)
                else:
                    total_result2.append(res)
                    total_result_modified2.append(res_mod)
                index += 1

    total_result_dict = {'epoch': epoch}
    total_result_dict['recall@5'] = get_metrics('recall@5', total_result)
    total_result_dict['recall@10'] = get_metrics('recall@10', total_result)
    total_result_dict['recall@20'] = get_metrics('recall@20', total_result)
    total_result_dict['mrr@5'] = get_metrics('mrr@5', total_result)
    total_result_dict['mrr@10'] = get_metrics('mrr@10', total_result)
    total_result_dict['mrr@20'] = get_metrics('mrr@20', total_result)
    total_result_dict['sum'] = total_result_dict['recall@5'] + total_result_dict['recall@10'] + \
                               total_result_dict['recall@20'] + total_result_dict['mrr@5'] + \
                               total_result_dict['mrr@10'] + total_result_dict['mrr@20']
    print(total_result_dict)

    total_result_dict_modified = {'epoch': epoch}
    total_result_dict_modified['recall@5'] = get_metrics('recall@5', total_result_modified)
    total_result_dict_modified['recall@10'] = get_metrics('recall@10', total_result_modified)
    total_result_dict_modified['recall@20'] = get_metrics('recall@20', total_result_modified)
    total_result_dict_modified['mrr@5'] = get_metrics('mrr@5', total_result_modified)
    total_result_dict_modified['mrr@10'] = get_metrics('mrr@10', total_result_modified)
    total_result_dict_modified['mrr@20'] = get_metrics('mrr@20', total_result_modified)
    total_result_dict_modified['sum'] = total_result_dict_modified['recall@5'] + total_result_dict_modified[
        'recall@10'] + total_result_dict_modified['recall@20'] + total_result_dict_modified['mrr@5'] + \
                                        total_result_dict_modified['mrr@10'] + total_result_dict_modified['mrr@20']
    print(total_result_dict_modified)

    total_result_dict2 = {'epoch': epoch}
    total_result_dict2['recall@5'] = get_metrics('recall@5', total_result2)
    total_result_dict2['recall@10'] = get_metrics('recall@10', total_result2)
    total_result_dict2['recall@20'] = get_metrics('recall@20', total_result2)
    total_result_dict2['mrr@5'] = get_metrics('mrr@5', total_result2)
    total_result_dict2['mrr@10'] = get_metrics('mrr@10', total_result2)
    total_result_dict2['mrr@20'] = get_metrics('mrr@20', total_result2)
    total_result_dict2['sum'] = total_result_dict2['recall@5'] + total_result_dict2['recall@10'] + \
                                total_result_dict2['recall@20'] + total_result_dict2['mrr@5'] + \
                                total_result_dict2['mrr@10'] + total_result_dict2['mrr@20']
    print(total_result_dict2)

    total_result_dict_modified2 = {'epoch': epoch}
    total_result_dict_modified2['recall@5'] = get_metrics('recall@5', total_result_modified2)
    total_result_dict_modified2['recall@10'] = get_metrics('recall@10', total_result_modified2)
    total_result_dict_modified2['recall@20'] = get_metrics('recall@20', total_result_modified2)
    total_result_dict_modified2['mrr@5'] = get_metrics('mrr@5', total_result_modified2)
    total_result_dict_modified2['mrr@10'] = get_metrics('mrr@10', total_result_modified2)
    total_result_dict_modified2['mrr@20'] = get_metrics('mrr@20', total_result_modified2)
    total_result_dict_modified2['sum'] = total_result_dict_modified2['recall@5'] + total_result_dict_modified2[
        'recall@10'] + total_result_dict_modified2['recall@20'] + total_result_dict_modified2['mrr@5'] + \
                                         total_result_dict_modified2['mrr@10'] + total_result_dict_modified2['mrr@20']
    print(total_result_dict_modified2)

    with open(save_path + 'evaluate_on_mod.txt', 'w') as f:
        for index, res, res_mod, seq, seq_mod in zip(indices, total_result, total_result_modified, seq_list,
                                                     seq_mod_list):
            result = {'index': index, 'seq_ori': seq, 'seq_mod': seq_mod, 'res_ori': res, 'res_mod': res_mod}
            f.write(str(result) + '\n')
        f.write(str(total_result_dict) + '\n')
        f.write(str(total_result_dict_modified) + '\n')
        f.write(str(total_result_dict2) + '\n')
        f.write(str(total_result_dict_modified2) + '\n')
