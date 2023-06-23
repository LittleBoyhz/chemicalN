import os
import pickle

import numpy as np
import pandas as pd
import xlrd2

import mdr_descriptors as mdr
import morgan_fingerprint as mfp
import morgan_pka as mpk
import ohe_pka as opk
import one_hot_encoding as ohe
from utils import split_dataset


class Arguments:
    def __init__(self):
        self.data_folder = "../datasets/real"
        self.dataset = "real_4"  # {"real_1", "real_2", "real_3"}
        self.sheet_name = "Sheet1"

        self.representation = "one_hot"  # {"one_hot", "morgan_fp"}
        self.morgan_fp_dims = [128, 256, 512, 1024, 2048]  # arguments for morgan fingerprint

        self.save = True


def main():
    args = Arguments()

    excel_path = os.path.join(args.data_folder, args.dataset, args.dataset + "_smiles_yields.xlsx")
    excel = xlrd2.open_workbook(excel_path)
    sheet = excel.sheet_by_name(args.sheet_name)

    if args.dataset in ["real_1", "real_4"]:
        num_columns_of_sheet = 5
    elif args.dataset in ["real_2"]:
        num_columns_of_sheet = 7
    elif args.dataset in ["real_3"]:
        num_columns_of_sheet = 4
    else:
        raise ValueError("Unknown dataset {}".format(args.dataset))

    save_folder = os.path.join(args.data_folder, args.dataset, args.dataset + "_" + args.representation)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    prefix = args.dataset + "_" + args.representation

    if args.representation == "one_hot":
        characters, max_length, char_to_idx, idx_to_char = ohe.get_chars_and_max_length(sheet, num_columns_of_sheet)
        encoding_of_dataset, flattened_encoding_of_dataset, yields = ohe.encode_dataset(sheet, char_to_idx, max_length, num_columns_of_sheet)

        smiles_seq = ohe.one_hot_to_smiles(flattened_encoding_of_dataset[-1, ...], idx_to_char, max_length)
        print(smiles_seq)

        if args.save:
            data_file = os.path.join(save_folder, prefix + ".npz")
            np.savez(data_file, data=flattened_encoding_of_dataset, labels=yields)
            with open(os.path.join(save_folder, prefix + "_metadata.pkl"), "wb") as f:
                pickle.dump(max_length, f)
                pickle.dump(char_to_idx, f)
                pickle.dump(idx_to_char, f)

            data = np.load(data_file)
            x, y = data["data"], data["labels"]
            (train_data, train_labels), (test_data, test_labels) = split_dataset(x, y)
            np.savez(os.path.join(save_folder, prefix + "_train.npz"),
                     train_data=train_data, train_labels=train_labels)
            np.savez(os.path.join(save_folder, prefix + "_test.npz"),
                     test_data=test_data, test_labels=test_labels)
    elif args.representation == "ohe_pka":
        characters, max_length, char_to_idx, idx_to_char = opk.get_chars_and_max_length(sheet, num_columns_of_sheet)
        encoding_of_dataset, flattened_encoding_of_dataset, yields = opk.encode_dataset(sheet, char_to_idx, max_length,
                                                                                        num_columns_of_sheet)

        smiles_seq = opk.one_hot_to_smiles(flattened_encoding_of_dataset[-1, ...], idx_to_char, max_length)  # why
        print(smiles_seq)

        if args.save:
            data_file = os.path.join(save_folder, prefix + ".npz")
            np.savez(data_file, data=flattened_encoding_of_dataset, labels=yields)
            with open(os.path.join(save_folder, prefix + "_metadata.pkl"), "wb") as f:  # why
                pickle.dump(max_length, f)  # +Serialization: converts a Python object hierarchy into a byte stream
                pickle.dump(char_to_idx, f)
                pickle.dump(idx_to_char, f)

            data = np.load(data_file)
            x, y = data["data"], data["labels"]
            (train_data, train_labels), (test_data, test_labels) = split_dataset(x, y)
            np.savez(os.path.join(save_folder, prefix + "_train.npz"),
                     train_data=train_data, train_labels=train_labels)
            np.savez(os.path.join(save_folder, prefix + "_test.npz"),
                     test_data=test_data, test_labels=test_labels)
    elif args.representaton == "morgan_fp":
        for dim_molecule in args.morgan_fp_dims:
            flattened_encoding_of_dataset, yields = mfp.encode_dataset(sheet, num_columns_of_sheet, dim_molecule)

            if args.save:
                curr_prefix = prefix + "_" + str(dim_molecule)
                data_file = os.path.join(save_folder, curr_prefix + ".npz")
                np.savez(data_file, data=flattened_encoding_of_dataset, labels=yields)

                data = np.load(data_file)
                x, y = data["data"], data["labels"]
                (train_data, train_labels), (test_data, test_labels) = split_dataset(x, y)
                np.savez(os.path.join(save_folder, curr_prefix + "_train.npz"),
                         train_data=train_data, train_labels=train_labels)
                np.savez(os.path.join(save_folder, curr_prefix + "_test.npz"),
                         test_data=test_data, test_labels=test_labels)
    elif args.representation == "morgan_pka":
        for dim_molecule in args.morgan_fp_dims:
            flattened_encoding_of_dataset, yields = mpk.encode_dataset(sheet, num_columns_of_sheet, dim_molecule)

            if args.save:
                curr_prefix = prefix + "_" + str(dim_molecule)
                data_file = os.path.join(save_folder, curr_prefix + ".npz")
                np.savez(data_file, data=flattened_encoding_of_dataset, labels=yields)
                # +Save several arrays (under corresponding names) into a single file in uncompressed .npz format

                data = np.load(data_file)
                x, y = data["data"], data["labels"]
                (train_data, train_labels), (test_data, test_labels) = split_dataset(x, y)
                np.savez(os.path.join(save_folder, curr_prefix + "_train.npz"),
                         train_data=train_data, train_labels=train_labels)
                np.savez(os.path.join(save_folder, curr_prefix + "_test.npz"),
                         test_data=test_data, test_labels=test_labels)
    elif args.representation == "Mordred":
        flattened_encoding_of_dataset, yields = mdr.encode_dataset(sheet, num_columns_of_sheet)

        if args.save:
            data_file = os.path.join(save_folder, prefix + ".npz")
            np.savez(data_file, data=flattened_encoding_of_dataset, labels=yields)
            # +Save several arrays (under corresponding names) into a single file in uncompressed .npz format

            data = np.load(data_file)
            x, y = data["data"], data["labels"]
            (train_data, train_labels), (test_data, test_labels) = split_dataset(x, y)
            np.savez(os.path.join(save_folder, prefix + "_train.npz"),
                     train_data=train_data, train_labels=train_labels)
            np.savez(os.path.join(save_folder, prefix + "_test.npz"),
                     test_data=test_data, test_labels=test_labels)
    else:
        raise ValueError("Unknown representaton {}".format(args.representation))


if __name__ == "__main__":
    main()
