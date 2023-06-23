import numpy as np
import util
import copy

use_product = True

smiles_s = util.get_smiles("./select_new.xlsx")
smiles_ffp = copy.copy(smiles_s)
dict_pkas, dict_bdes = util.get_pka_bde("./pka_bde_01_p.xlsx")
if use_product:
    products_dict, products_smile = util.get_products("./products.xlsx")
    smiles_ffp.append(products_smile)
fps = util.get_fps(smiles_ffp)

with open("./test.csv","w") as f:
    past = []
    all_fp = []
    all_pka = []
    def dfs(place):
        if place == 7:
            if use_product:
                chem1 = past[1]
                chem2 = past[2]
                product = products_dict[(chem1,chem2)]
                past.append(product)
            ans = ""
            fp = []
            pka_bde = []
            for smile in past:
                ans = ans + smile + ','
                fp.append(fps[smile])
                pka_bde.append(dict_pkas[smile])
                pka_bde.append(dict_bdes[smile])
            pka_bde = np.array(pka_bde)
            fp = np.concatenate(fp)
            all_fp.append(fp)
            all_pka.append(pka_bde)
            f.write(ans + '\n')
            if use_product:
                past.pop()
            return
        
        smiles = smiles_s[place]
        
        for smile in smiles:
            past.append(smile)
            dfs(place + 1)
            past.pop()
    
    dfs(0)
    all_fp = np.array(all_fp)
    all_pka = np.array(all_pka)
    yields = np.zeros(shape=(1,all_fp.shape[0]))
    np.savez("../datasets/real/real_6/split_0/real_6_morgan_fp/real_6_morgan_fp_2048_test.npz", test_data=all_fp, test_labels=yields)
    np.savez("../datasets/real/real_6/split_0/real_6_pka_bde01/real_6_pka_bde01_test.npz", test_data=all_pka, test_labels=yields)