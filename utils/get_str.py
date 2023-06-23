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
import myrxnfp

class Arguments:
    def __init__(self, split_mode, representation):
        self.data_folder = "../datasets/real"
        self.dataset = "real_4"
        self.sheet_name = "Sheet1"

        self.split_mode = split_mode  # {1, 2, 3, 4, 5}

        self.representation = representation  # {"one_hot", "morgan_fp", "Mordred", "morgan_pka", "ohe_pka"}
        self.morgan_fp_dims = [2048]  # arguments for morgan fingerprint

        self.save = True

def main():
    strs = input("filename")
    type = input("test or train")
    sms = [0]
    reprs = ["pka_bde01", "morgan_fp", "Mordred", "rxnfp"]

    for split_mode in sms:
        for representation in reprs:
            print("split_mode: {}, representation: {}".format(split_mode, representation))

            args = Arguments(split_mode, representation)
            excel_path = os.path.join(args.data_folder, args.dataset, args.dataset + "_smiles_yields" + strs + ".xlsx")
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
            if type == "test":
                if args.representation == "Mordred":
                    flattened_encoding_of_dataset, yields = mdr_descriptors.encode_dataset(sheet, num_columns_of_sheet)
                    if args.save:
                        np.savez(os.path.join(save_folder, prefix + "_test.npz"),
                                test_data=flattened_encoding_of_dataset, test_labels=yields)
                elif args.representation == "morgan_fp":
                    for dim_molecule in args.morgan_fp_dims:
                        flattened_encoding_of_dataset, yields = mfp.encode_dataset(sheet, num_columns_of_sheet, dim_molecule)
                        if args.save:
                            curr_prefix = prefix + "_" + str(dim_molecule)
                            data_file = os.path.join(save_folder, curr_prefix + "_test.npz")
                            np.savez(data_file, test_data=flattened_encoding_of_dataset, test_labels=yields)
                elif args.representation == "pka_bde01":
                    flattened_encoding_of_dataset, yields = pka_bde.encode_dataset(sheet, num_columns_of_sheet, args, True, 6)
                    if args.save:
                        data_file = os.path.join(save_folder, prefix + "_test.npz")
                        np.savez(data_file, test_data=flattened_encoding_of_dataset, test_labels=yields)
                elif args.representation == "rxnfp":
                    flattened_encoding_of_dataset, yields = myrxnfp.encode_dataset(sheet, num_columns_of_sheet)
                    if args.save:
                        data_file = os.path.join(save_folder, prefix + "_test.npz")
                        np.savez(data_file, test_data=flattened_encoding_of_dataset, test_labels=yields)
                else:
                    raise ValueError("Unknown representaton {}".format(args.representation))
            else:
                if args.representation == "Mordred":
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
                elif args.representation == "pka_bde01":
                    flattened_encoding_of_dataset, yields = pka_bde.encode_dataset(sheet, num_columns_of_sheet, args, True, 6)
                    if args.save:
                        data_file = os.path.join(save_folder, prefix + "_train.npz")
                        np.savez(data_file, train_data=flattened_encoding_of_dataset, train_labels=yields)
                elif args.representation == "rxnfp":
                    flattened_encoding_of_dataset, yields = myrxnfp.encode_dataset(sheet, num_columns_of_sheet)
                    if args.save:
                        data_file = os.path.join(save_folder, prefix + "_train.npz")
                        np.savez(data_file, train_data=flattened_encoding_of_dataset, train_labels=yields)
                else:
                    raise ValueError("Unknown representaton {}".format(args.representation))

            print("\n")

if __name__ == "__main__":
    main()