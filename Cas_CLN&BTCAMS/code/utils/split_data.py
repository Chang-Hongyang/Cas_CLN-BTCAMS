import json
import os
import random


def split_data(data_path):
    res_data = []
    with open(data_path, 'r') as fr:
        for line in fr.readlines():
            data_line = json.loads(line.strip())
            res_data.append(data_line)
    return res_data


def generate_data(data_dir, data_set, flag):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    with open(data_dir + '/{}_data.json'.format(flag), 'w') as fw:
        for data in data_set:
            fw.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    train_data = split_data('../../raw_data/train_data.json')
    val_data = split_data('../../raw_data/val_data.json')
    print(len(train_data), len(val_data))
    print()
    total_data = train_data + val_data
    random.seed(194012)
    random.shuffle(train_data)
    num=1132

    for i in range(4):
        val_ = train_data[i * num:(i + 1) * num]
        train_1 = train_data[:i * num] + train_data[(i + 1) * num:]
        train_ = train_1 + val_data
        print(len(train_), len(val_))
        print()
        generate_data('../../user_data/split_data/data_set_{}'.format(i + 1), train_, 'train')
        generate_data('../../user_data/split_data/data_set_{}'.format(i + 1), val_, 'val')

    # 注意⚠️：第5份data_set_5 是主办方原始给到的训练+测试集
