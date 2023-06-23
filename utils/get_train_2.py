import os
import pickle

import numpy as np
import pandas
import pandas as pd
import xlrd2

import morgan_fingerprint as mfp
import one_hot_encoding as ohe

import utils
import random
import pka_bde
import mdr_descriptors

def split(x, y, train_idx, test_idx):
    train_data, train_labels = x[train_idx, ...], y[train_idx, ...]
    test_data, test_labels = x[test_idx, ...], y[test_idx, ...]
    
    print(train_labels.min())
    print(train_labels.max())
    print(test_labels.min())
    print(test_labels.max())

    return (train_data, train_labels), (test_data, test_labels)


class Arguments:
    def __init__(self, split_mode, representation):
        self.data_folder = "../datasets/real"
        self.dataset = "real_4"
        self.sheet_name = "Sheet1"

        self.split_mode = split_mode  # {1, 2, 3, 4, 5}

        self.representation = representation  # {"one_hot", "morgan_fp", "Mordred", "morgan_pka", "ohe_pka"}
        self.morgan_fp_dims = [2048]  # arguments for morgan fingerprint

        self.save = True

def random_dataset(label_num, ratio_test=0.3):
    test_num = int(label_num * ratio_test)
    test_list = random.sample(range(0, label_num), test_num)
    test_list=list(set(test_list))
    train_list = []
    for i in range(0,label_num):
        if i not in test_list:
            train_list.append(i)
    return np.array(train_list),np.array(test_list)

def main():
    sms = [0]
    reprs = ["pka_bde01", "morgan_fp", "Mordred"]

    for split_mode in sms:
        for representation in reprs:
            print("split_mode: {}, representation: {}".format(split_mode, representation))

            args = Arguments(split_mode, representation)
            excel_path = os.path.join(args.data_folder, args.dataset, args.dataset + "_smiles_yields_product_2_train_2.xlsx")
            excel = xlrd2.open_workbook(excel_path)
            sheet = excel.sheet_by_name(args.sheet_name)

            if args.dataset in ["real_1", "real_4"]:
                num_columns_of_sheet = 5
            elif args.dataset in ["real_2", "real_5"]:
                num_columns_of_sheet = 7
            elif args.dataset in ["real_3"]:
                num_columns_of_sheet = 4
            elif args.dataset in ["real_6"]:
                num_columns_of_sheet = 9    # product
            else:
                raise ValueError("Unknown dataset {}".format(args.dataset))

            save_folder = os.path.join(args.data_folder, args.dataset, "split_" + str(args.split_mode),
                                       args.dataset + "_" + args.representation)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            prefix = args.dataset + "_" + args.representation

            if args.representation == "one_hot":
                characters, max_length, char_to_idx, idx_to_char = ohe.get_chars_and_max_length(sheet, num_columns_of_sheet)
                encoding_of_dataset, flattened_encoding_of_dataset, yields = ohe.encode_dataset(sheet, char_to_idx, max_length, num_columns_of_sheet)
                smiles_seq = ohe.one_hot_to_smiles(flattened_encoding_of_dataset[-1, ...], idx_to_char, max_length)
                print(smiles_seq)
                if args.save:
                    data_file = os.path.join(save_folder, prefix + "_train.npz")
                    np.savez(data_file, train_data=flattened_encoding_of_dataset, train_labels=yields)
                    with open(os.path.join(save_folder, prefix + "_metadata.pkl"), "wb") as f:
                        pickle.dump(max_length, f)
                        pickle.dump(char_to_idx, f)
                        pickle.dump(idx_to_char, f)
            elif args.representation == "Mordred":
                flattened_encoding_of_dataset, yields = mdr_descriptors.encode_dataset(sheet, num_columns_of_sheet)
                if args.save:
                    np.savez(os.path.join(save_folder, prefix + "_train.npz"),
                             train_data=flattened_encoding_of_dataset, train_labels=yields)
            elif args.representation == "morgan_fp":
                for dim_molecule in args.morgan_fp_dims:
                    flattened_encoding_of_dataset, yields = mfp.encode_dataset(sheet, num_columns_of_sheet, dim_molecule)
                    if args.save:
                        curr_prefix = prefix + "_" + str(dim_molecule)
                        data_file = os.path.join(save_folder, curr_prefix + "_train.npz")
                        np.savez(data_file, train_data=flattened_encoding_of_dataset, train_labels=yields)
            elif args.representation == "pka_bde":
                flattened_encoding_of_dataset, yields = pka_bde.encode_dataset(sheet, num_columns_of_sheet, args, False, 5)
                if args.save:
                    data_file = os.path.join(save_folder, prefix + "_train.npz")
                    np.savez(data_file, train_data=flattened_encoding_of_dataset, train_labels=yields)
            elif args.representation == "pka_bde01":
                flattened_encoding_of_dataset, yields = pka_bde.encode_dataset(sheet, num_columns_of_sheet, args, True, 6)
                if args.save:
                    data_file = os.path.join(save_folder, prefix + "_train.npz")
                    np.savez(data_file, train_data=flattened_encoding_of_dataset, train_labels=yields)

            else:
                raise ValueError("Unknown representaton {}".format(args.representation))

            print("\n")


if __name__ == "__main__":
    main()