import xlrd2
import numpy as np
import util
import copy

smiles_s = util.get_smiles("./select_new.xlsx")
smiles_ffp = copy.copy(smiles_s)
dict_pkas, dict_bdes = util.get_pka_bde("./pka_bde_01_p.xlsx")
products_dict, products_smile = util.get_products("./products.xlsx")
smiles_ffp.append(products_smile)
fps = util.get_fps(smiles_ffp)

save_test = True

exx_reaction = xlrd2.open_workbook("./low_label.xlsx")
work_sheet = exx_reaction.sheet_by_index(0)
nrows = work_sheet.nrows
ncol = work_sheet.ncols
all_fp = []
all_pka = []
all_yield = []
for i in range(1,nrows):
    fp = []
    pka_bde = []
    for j in range(0,ncol-1):
        smile = work_sheet.row_values(i)[j]
        fp.append(fps[smile])
        pka_bde.append(dict_pkas[smile])
        pka_bde.append(dict_bdes[smile])
    pka_bde = np.array(pka_bde)
    fp = np.concatenate(fp)
    all_fp.append(fp)
    all_pka.append(pka_bde)
    all_yield.append(work_sheet.row_values(i)[ncol-1])
    
all_fp = np.array(all_fp)
all_pka = np.array(all_pka)
yields = np.array(all_yield)
if save_test:
    np.savez("../datasets/real/real_6/split_0/real_6_morgan_fp/real_6_morgan_fp_2048_test.npz", test_data=all_fp, test_labels=yields)
    np.savez("../datasets/real/real_6/split_0/real_6_pka_bde01/real_6_pka_bde01_test.npz", test_data=all_pka, test_labels=yields)
else:
    np.savez("../datasets/real/real_6/split_0/real_6_morgan_fp/real_6_morgan_fp_2048_train.npz", train_data=all_fp, train_labels=yields)
    np.savez("../datasets/real/real_6/split_0/real_6_pka_bde01/real_6_pka_bde01_train.npz", train_data=all_pka, train_labels=yields)