def classfi(seq_len):
    length_lower_bound = [0, 20, 30, 40]
    length_upper_bound = [20, 30, 40, 51]
    len_number = [0, 0, 0, 0]
    for j in range(len(length_lower_bound)):
        # print(j)
        metrics = {}
        filter_dataobject = []
        for i in range(len(seq_len)):  # length filter
            length_seq = seq_len[i]  # start at 1
            if length_lower_bound[j] <= length_seq < length_upper_bound[j]:
                len_number[j] += 1
    all_number = len(seq_len)
    print("all  " + str(all_number))
    for j in range(len(length_lower_bound)):
        print(j)
        len_pro = float(len_number[j]) / float(all_number) * 100  # proportion
        print('%d | %.4f' % (len_number[j], len_pro))
        # print(len_pro)


def length(User):
    length_list = []
    for user in User:
        length_list.append(len(User[user]) - 1)
    classfi(length_list)