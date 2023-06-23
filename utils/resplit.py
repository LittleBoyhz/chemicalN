import numpy as np
import random

def split(x, y, train_idx, test_idx):
    train_data, train_labels = x[train_idx, ...], y[train_idx, ...]
    test_data, test_labels = x[test_idx, ...], y[test_idx, ...]

    print(train_labels.min())
    print(train_labels.max())
    print(test_labels.min())
    print(test_labels.max())

    return (train_data, train_labels), (test_data, test_labels)

def random_dataset(label_num, ratio_test=0.2):
    test_num = int(label_num * ratio_test)
    test_list = random.sample(range(0, label_num), test_num)
    test_list=list(set(test_list))
    train_list = []
    for i in range(0,label_num):
        if i not in test_list:
            train_list.append(i)
    return np.array(train_list),np.array(test_list)

path=r"../datasets/real/real_5/puredata/"
data_files = ["real_5_morgan_fp_2048", "real_5_rxnfp", "real_5_pka_bde", "real_5_pka_bde01"]
train_idx, test_idx = random_dataset(5760, 0.5)
for data_file in data_files:
    data = np.load(path + data_file + ".npz")
    x, y = data["data"], data["labels"]
    (train_data, train_labels), (test_data, test_labels) = split(x, y, train_idx, test_idx)
    np.savez(data_file + "_train.npz",
                train_data=train_data, train_labels=train_labels)
    np.savez(data_file + "_test.npz",
                test_data=test_data, test_labels=test_labels)