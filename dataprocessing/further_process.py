import random
from tqdm import tqdm


def train_process(train_file, output_train_file):
    max_len = 0
    with open(train_file, 'r', encoding='utf-8') as f1, open(output_train_file, 'a+', encoding='utf-8') as out_file1:
        current_user_id = 0
        train_seq = []
        for line1 in tqdm(f1):
            feature = line1.strip().split('\t')
            if int(feature[0]) == current_user_id:
                train_seq.append(int(feature[1]) + 1)
            else:
                current_user_id += 1
                out_file1.write(','.join(map(str, train_seq)) + '\n')
                max_len = max(max_len, len(train_seq))
                train_seq.clear()
                train_seq.append(int(feature[1]) + 1)  
    print(max_len)
    return


def valid_process(out_train, input_valid, out_valid):
    with open(out_train, 'r', encoding='utf-8') as f1, open(input_valid, 'r', encoding='utf-8') as f2, open(out_valid,
                                                                                                            'a+',
                                                                                                            encoding='utf-8') as f3:
        for line1 in tqdm(f1):
            valid_seq = []
            line2 = f2.readline()
            item_seq = line1.strip().split(',')
            valid_seq += item_seq
            valid_item = line2.split('\t')
            valid_seq.append(int(valid_item[1]) + 1)
            f3.write(','.join(map(str, valid_seq)) + '\n')
        return


def test_process(out_valid, input_test, out_test):
    max_item = 0
    with open(out_valid, 'r', encoding='utf-8') as f1, open(input_test, 'r', encoding='utf-8') as f2, open(out_test,
                                                                                                           'a+',
                                                                                                           encoding='utf-8') as f3:
        for line1 in tqdm(f1):
            test_seq = []
            line2 = f2.readline()
            item_seq = line1.strip().split(',')
            test_seq += item_seq
            item_seq = (int(i) for i in item_seq)
            max_item = max(max_item, max(item_seq))
            test_item = line2.split('\t')
            test_seq.append(int(test_item[1]) + 1)
            max_item = max(max_item, int(test_item[1]) + 1)
            f3.write(','.join(map(str, test_seq)) + '\n')
    print(max_item)
    return


def negative_generator(input_file, output_file, dataset):
    with open(input_file, 'r', encoding='utf-8') as f1, open(output_file, 'a+', encoding='utf-8') as f2:
        for line in tqdm(f1):
            if dataset == 'beauty':
                item_num = 12102 # include padding, exclude eos and mask
            if dataset == 'sports':
                item_num = 18358                
            if dataset == 'yelp':
                item_num = 16553
            item_set = set(i for i in range(1, item_num))
            exist_item_id = line.strip().split(',')
            for i in exist_item_id:
                if int(i) in item_set:
                    item_set.remove(int(i))
            neg_item = random.sample(item_set, 99)
            f2.write(','.join(map(str, neg_item)) + '\n')
    return


if __name__ == '__main__':
    train_process('./Beauty/train.txt', './Beauty/train.dat')
    valid_process('./Beauty/train.dat', './Beauty/valid.txt', './Beauty/valid.dat')
    test_process('./Beauty/valid.dat', './Beauty/test.txt', './Beauty/test.dat')
    negative_generator('./Beauty/valid.dat', './Beauty/valid_neg.dat', 'beauty')
    negative_generator('./Beauty/test.dat', './Beauty/test_neg.dat', 'beauty')
