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
from script import init_seeds, xavier_init, seqs_normalization, seqs_normalization_test, evaluate_function, get_metrics, cal_similarity, stat_operation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-tf', type=str, default='./Beauty/train.dat', help='path of training dataset')
    parser.add_argument('-vf', type=str, default='./Beauty/valid.dat', help='path of valid dataset')
    parser.add_argument('-ef', type=str, default='./Beauty/test.dat', help='path of test dataset')
    parser.add_argument('-vn', type=str, default='./Beauty/valid_neg.dat', help='path of valid dataset(negative instance)')
    parser.add_argument('-en', type=str, default='./Beauty/test_neg.dat', help='path of test dataset(negative instance)')
    parser.add_argument('-b', type=int, default=256, help='batch_size')
    parser.add_argument('-ls', type=int, default=50, help='log_step')
    parser.add_argument('-l', type=float, default=1e-3, help='learning_rate')
    parser.add_argument('-e', type=int, default=300, help='epoch_num')
    parser.add_argument('-dr', type=float, default=0.5, help='normalization')
    parser.add_argument('-hd', type=int, default=64, help='hidden_layer_dimension')
    parser.add_argument('-hn', type=int, default=1, help='head_num')
    parser.add_argument('-ln', type=int, default=1, help='number_of_transformer_layer')
    parser.add_argument('-o', type=str, default='./save_model/', help='save_path')
    parser.add_argument('-m', type=str, default="train", help='train_valid_test')
    parser.add_argument('-r', action='store_true', help='resume')
    parser.add_argument('-n', type=int, default=12104, help='item_num')  # Beauty dataset contains 12101 items, plus padding, EOS, mask
    parser.add_argument('-ml', type=int, default=50, help='max_seqs_len')
    parser.add_argument('-mml', type=int, default=60, help='modified_max_seqs_len')
    parser.add_argument('-mi', type=int, default=5, help='max_insert_num')
    parser.add_argument('-mb', type=float, default=0.5, help='mask_prob')
    parser.add_argument('-p', type=float, nargs='+', default=[0.4, 0.9], help='plist')  # for correction tasks, <= plist[0] -> keep, plist[0]<&<=plist[1] -> delete, plist[1]< -> insert
    args = parser.parse_args()
    train_file = args.tf
    valid_file = args.vf
    test_file = args.ef
    valid_neg_file = args.vn
    test_neg_file = args.en
    batch_size = args.b
    log_step = args.ls
    learning_rate = args.l
    epochs = args.e
    dropout_rate = args.dr
    hidden_unit = args.hd
    head_num = args.hn
    layer_num = args.ln
    save_path = args.o
    mode = args.m
    resume = args.r
    item_num = args.n
    max_seqs_len = args.ml
    modified_max_seqs_len = args.mml
    max_insert_num = args.mi
    mask_prob = args.mb
    plist = args.p

    init_seeds()

    if mode == 'train':
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path + 'model/'):
            os.makedirs(save_path + 'model/')
        if resume:
            fw = open(save_path + 'train_result.txt', 'a')
        else:
            fw = open(save_path + 'train_result.txt', 'w')
        dataset = TrainDataset(train_file, item_num, max_seqs_len, modified_max_seqs_len, max_insert_num, mask_prob, plist)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        model = model(dropout_rate, hidden_unit, item_num, head_num, layer_num, max_insert_num)

        last_epoch = 0
        if resume:
            with open(save_path + 'train_result.txt', 'r') as f:
                content = f.readlines()
            last_epoch = int(len(content) / 2) - 1
            print('load model:Epoch %d' % (last_epoch,))
            model.load_state_dict(torch.load(save_path + 'model/steam-' + str(last_epoch) + '.pth'))
        else:
            print('initialize model')
            model.apply(xavier_init)
        if torch.cuda.is_available():
            print("cuda is available")
            model.cuda()
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        epoch = last_epoch
        while epoch < epochs:
            epoch += 1
            step = 0
            acc_loss_corrector = 0
            acc_loss_recommender = 0
            start_time = time.time()
            for batch in dataloader:
                step += 1
                optimizer.zero_grad()
                seq, masked_seq, random_modified_seqs, l1_ground_truth, l2_ground_truth, insert_seqs, rec_loss_mask = batch
                if torch.cuda.is_available():
                    seq = seq.cuda()
                    masked_seq = masked_seq.cuda()
                    random_modified_seqs = random_modified_seqs.cuda()
                    l1_ground_truth = l1_ground_truth.cuda()
                    l2_ground_truth = l2_ground_truth.cuda()
                    insert_seqs = insert_seqs.cuda()
                    rec_loss_mask = rec_loss_mask.cuda()
                """
                Through this function, we can get the modified sequence prediction. That is, for each time step,
                whether to keep, delete, or insert
                """
                full_layer_output, insert_net_output, padding_mask = model.corrector_forward(
                    random_modified_seqs, insert_seqs)
                l1_loss, l2_loss = model.corrector_loss(full_layer_output, insert_net_output,
                                                        l1_ground_truth, l2_ground_truth,
                                                        padding_mask)
                loss1 = l1_loss.sum() / (random_modified_seqs != 0).sum()
                if (l2_ground_truth != 0).sum() == 0:
                    loss2 = l2_loss.sum()
                else:
                    loss2 = l2_loss.sum() / (l2_ground_truth != 0).sum()
                total_loss = loss1 + loss2
                acc_loss_corrector += total_loss
                if step % log_step == 0:
                    print('Epoch %d Step %d corrector loss %0.4f Time %d' % (
                        epoch, step, acc_loss_corrector / step, time.time() - start_time))
                model.eval()
                with torch.no_grad():
                    # Predict the correction operation for each time step of the original sequence
                    full_layer_output_ori, insert_net_output_ori = model.corrector_inference(seq)
                    # Based on above prediction, correct the original sequence
                    modified_seqs = model.seqs_correction(full_layer_output_ori, insert_net_output_ori, seq)
                    # Process the corrected sequence for the masked item prediction task
                    modified_masked_seqs, modified_rec_loss_mask, modified_ori_seqs = seqs_normalization(
                        modified_seqs, modified_max_seqs_len, item_num, mask_prob)
                model.train()
                if torch.cuda.is_available():
                    modified_masked_seqs = modified_masked_seqs.cuda()
                    modified_rec_loss_mask = modified_rec_loss_mask.cuda()
                    modified_ori_seqs = modified_ori_seqs.cuda()
                # Bring the original sequence into recommender
                recommender_output = model.forward(masked_seq)
                recommender_loss = model.recommender_loss(recommender_output, rec_loss_mask, seq)
                if (rec_loss_mask != 0).sum() == 0:
                    rec_loss = recommender_loss.sum()
                else:
                    rec_loss = recommender_loss.sum() / (rec_loss_mask != 0).sum()
                total_recommender_loss = rec_loss
                # Bring the modified sequence into recommender
                modified_recommender_output = model.forward(modified_masked_seqs)
                modified_recommender_loss = model.recommender_loss(modified_recommender_output,
                                                                   modified_rec_loss_mask,
                                                                   modified_ori_seqs)
                if (modified_rec_loss_mask != 0).sum() == 0:
                    modified_rec_loss = modified_recommender_loss.sum()
                else:
                    modified_rec_loss = modified_recommender_loss.sum() / (modified_rec_loss_mask!=0).sum()
                total_recommender_loss += modified_rec_loss
                acc_loss_recommender += total_recommender_loss
                loss = total_loss+total_recommender_loss
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                if step % log_step == 0:
                    print('Epoch %d Step %d recommender loss %0.4f Time %d' % (
                        epoch, step, acc_loss_recommender / step, time.time() - start_time))
            torch.save(model.state_dict(), save_path + 'model/steam-' + str(epoch) + '.pth')
            print(
                'Epoch %d corrector loss %0.4f Time %d' % (epoch, acc_loss_corrector / step, time.time() - start_time))
            print('Epoch %d recommender loss %0.4f Time %d' % (
                epoch, acc_loss_recommender / step, time.time() - start_time))
            fw.write('Epoch %d corrector loss %0.4f' % (epoch, acc_loss_corrector / step) + '\n')
            fw.write('Epoch %d recommender loss %0.4f' % (epoch, acc_loss_recommender / step) + '\n')
            fw.flush()
        fw.close()

    if mode == "valid":
        if resume:
            fw = open(save_path + 'valid_result.txt', 'a')
            fw2 = open(save_path + 'valid_result_modified.txt', 'a')
        else:
            fw = open(save_path + 'valid_result.txt', 'w')
            fw2 = open(save_path + 'valid_result_modified.txt', 'w')
        dataset = ValidDataset(valid_file, valid_neg_file, item_num, modified_max_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        model = model(dropout_rate, hidden_unit, item_num, head_num, layer_num, max_insert_num)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        next_epoch = 1
        if resume:
            with open(save_path + 'valid_result_modified.txt', 'r') as f:
                content = f.readlines()
            next_epoch = len(content) + 1
            print('valid from epoch %d' % (next_epoch,))
        epoch = next_epoch
        while epoch <= epochs:
            step = 0
            total_result = []
            total_result_modified = []
            model.load_state_dict(torch.load(save_path + 'model/steam-' + str(epoch) + '.pth'))
            with torch.no_grad():
                for batch in dataloader:
                    step += 1
                    _, valid_neg, masked_seq, leave_one_seq, target = batch
                    if torch.cuda.is_available():
                        valid_neg = valid_neg.cuda()
                        masked_seq = masked_seq.cuda()
                        leave_one_seq = leave_one_seq.cuda()
                        target = target.cuda()
                    # original sequence
                    recommender_output = model.forward(masked_seq)
                    recommender_output = recommender_output[masked_seq == item_num - 1]
                    result = evaluate_function(recommender_output, target, valid_neg)
                    total_result.extend(result)
                    # corrected sequence
                    full_layer_output_ori, insert_net_output_ori = model.corrector_inference(leave_one_seq)
                    modified_leave_one_seqs = model.seqs_correction(full_layer_output_ori, insert_net_output_ori,leave_one_seq)
                    modified_masked_seqs = seqs_normalization_test(modified_leave_one_seqs, modified_max_seqs_len, item_num)
                    if torch.cuda.is_available():
                        modified_masked_seqs = modified_masked_seqs.cuda()
                    recommender_modified_output = model.forward(modified_masked_seqs)
                    recommender_modified_output = recommender_modified_output[modified_masked_seqs == item_num - 1]
                    result_modified = evaluate_function(recommender_modified_output, target, valid_neg)
                    total_result_modified.extend(result_modified)

            # metrics of original sequence
            total_result_dict = {'epoch': epoch}
            total_result_dict['recall@5'] = get_metrics('recall@5', total_result)
            total_result_dict['recall@10'] = get_metrics('recall@10', total_result)
            total_result_dict['recall@20'] = get_metrics('recall@20', total_result)
            total_result_dict['mrr@5'] = get_metrics('mrr@5',total_result)
            total_result_dict['mrr@10'] = get_metrics('mrr@10', total_result)
            total_result_dict['mrr@20'] = get_metrics('mrr@20', total_result)
            total_result_dict['sum'] = total_result_dict['recall@5'] + total_result_dict['recall@10'] + \
                                            total_result_dict['recall@20'] + total_result_dict['mrr@5'] +\
                                                total_result_dict['mrr@10'] + total_result_dict['mrr@20']
            print(total_result_dict)
            fw.write(str(total_result_dict) + '\n')
            # metrics of corrected sequence
            total_result_dict_modified = {'epoch': epoch}
            total_result_dict_modified['recall@5'] = get_metrics('recall@5', total_result_modified)
            total_result_dict_modified['recall@10'] = get_metrics('recall@10',total_result_modified)
            total_result_dict_modified['recall@20'] = get_metrics('recall@20', total_result_modified)
            total_result_dict_modified['mrr@5'] = get_metrics('mrr@5', total_result_modified)
            total_result_dict_modified['mrr@10'] = get_metrics('mrr@10', total_result_modified)
            total_result_dict_modified['mrr@20'] = get_metrics('mrr@20', total_result_modified)
            total_result_dict_modified['sum'] = total_result_dict_modified['recall@5'] + total_result_dict_modified[
                'recall@10'] + total_result_dict_modified['recall@20'] + total_result_dict_modified['mrr@5']+ \
                total_result_dict_modified['mrr@10'] + total_result_dict_modified ['mrr@20']
            print(total_result_dict_modified)
            fw2.write(str(total_result_dict_modified) + '\n')

            with open(save_path + 'valid_result_' + str(epoch) + '.txt', 'w') as f:
                for result in total_result:
                    f.write(str(result) + '\n')

            with open(save_path + 'valid_result_modified_' + str(epoch) + '.txt', 'w') as f2:
                for result in total_result_modified:
                    f2.write(str(result) + '\n')
            epoch += 1
        fw.close()
        fw2.close()

    if mode == "test":
        if resume:
            fw = open(save_path + 'test_result.txt', 'a')
            fw2 = open(save_path + 'test_result_modified.txt', 'a')
            fw3 = open(save_path + 'similarity.txt', 'a')	
        else:
            fw = open(save_path + 'test_result.txt', 'w')
            fw2 = open(save_path + 'test_result_modified.txt', 'w')
            fw3 = open(save_path + 'similarity.txt', 'w')
        dataset = TestDataset(test_file, test_neg_file, item_num, modified_max_seqs_len)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        model = model(dropout_rate, hidden_unit, item_num, head_num, layer_num, max_insert_num)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        next_epoch = 1
        if resume:
            with open(save_path + 'test_result_modified.txt', 'r') as f:
                content = f.readlines()
            next_epoch = len(content) + 1
            print('test from epoch %d' % (next_epoch,))
        epoch = next_epoch
        while epoch <= epochs:
            similarity_list = []
            operation_list = []
            step = 0
            total_result = []
            total_result_modified = []
            model.load_state_dict(torch.load(save_path + 'model/steam-' + str(epoch) + '.pth'))
            with torch.no_grad():
                for batch in dataloader:
                    step += 1
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
                    total_result.extend(result)
                    # corrected sequence
                    full_layer_output_ori, insert_net_output_ori = model.corrector_inference(leave_one_seq)
                    modified_leave_one_seqs = model.seqs_correction(full_layer_output_ori, insert_net_output_ori,leave_one_seq)
                    modified_masked_seqs = seqs_normalization_test(modified_leave_one_seqs, modified_max_seqs_len, item_num)
                    if torch.cuda.is_available():
                        modified_masked_seqs = modified_masked_seqs.cuda()
                    recommender_modified_output = model.forward(modified_masked_seqs)
                    recommender_modified_output = recommender_modified_output[modified_masked_seqs == item_num - 1]
                    result_modified = evaluate_function(recommender_modified_output, target, test_neg)
                    total_result_modified.extend(result_modified)
                    similarity = cal_similarity(masked_seq, modified_masked_seqs)
                    similarity_list.extend(similarity)
                    operation = stat_operation(full_layer_output_ori, leave_one_seq)
                    operation_list.extend(operation)
            # metrics of original sequence
            total_result_dict = {'epoch': epoch}
            total_result_dict['recall@5'] = get_metrics('recall@5', total_result)
            total_result_dict['recall@10'] = get_metrics('recall@10', total_result)
            total_result_dict['recall@20'] = get_metrics('recall@20', total_result)
            total_result_dict['mrr@5'] = get_metrics('mrr@5',total_result)
            total_result_dict['mrr@10'] = get_metrics('mrr@10', total_result)
            total_result_dict['mrr@20'] = get_metrics('mrr@20', total_result)
            total_result_dict['sum'] = total_result_dict['recall@5'] + total_result_dict['recall@10'] + \
                                            total_result_dict['recall@20'] + total_result_dict['mrr@5'] +\
                                                total_result_dict['mrr@10'] + total_result_dict['mrr@20']
            print(total_result_dict)
            fw.write(str(total_result_dict) + '\n')
            # metrics of corrected sequence
            total_result_dict_modified = {'epoch': epoch}
            total_result_dict_modified['recall@5'] = get_metrics('recall@5', total_result_modified)
            total_result_dict_modified['recall@10'] = get_metrics('recall@10',total_result_modified)
            total_result_dict_modified['recall@20'] = get_metrics('recall@20', total_result_modified)
            total_result_dict_modified['mrr@5'] = get_metrics('mrr@5', total_result_modified)
            total_result_dict_modified['mrr@10'] = get_metrics('mrr@10', total_result_modified)
            total_result_dict_modified['mrr@20'] = get_metrics('mrr@20', total_result_modified)
            total_result_dict_modified['sum'] = total_result_dict_modified['recall@5'] + total_result_dict_modified[
                'recall@10'] + total_result_dict_modified['recall@20'] + total_result_dict_modified['mrr@5'] + \
                total_result_dict_modified['mrr@10'] + total_result_dict_modified ['mrr@20']
            print(total_result_dict_modified)
            fw2.write(str(total_result_dict_modified) + '\n')

            operation_list = np.array(operation_list)
            operation_list = operation_list.sum(0)/operation_list.sum()
            similarity_result = {'similarity': sum(similarity_list) / len(similarity_list), 'operation': operation_list.tolist()}
            print(str(similarity_result))
            fw3.write(str(similarity_result) + '\n')

            with open(save_path + 'test_result_' + str(epoch) + '.txt', 'w') as f:
                for result in total_result:
                    f.write(str(result) + '\n')

            with open(save_path + 'test_result_modified_' + str(epoch) + '.txt', 'w') as f2:
                for result in total_result_modified:
                    f2.write(str(result) + '\n')

            epoch += 1
        fw.close()
        fw2.close()
        fw3.close()
