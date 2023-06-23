import xlrd2
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def get_pka_bde(filename):
    dict_pkas = {}
    dict_bdes = {}
    exx_pka = xlrd2.open_workbook(filename)
    sheet_names = exx_pka.sheet_names()
    num_sheets = len(sheet_names)
    for i in range(num_sheets):
        work_sheet = exx_pka.sheet_by_index(i)
        nrows = work_sheet.nrows
        ncol = work_sheet.ncols
        for j in range(1,nrows):
            smile = work_sheet.row_values(j)[0]
            pka = work_sheet.row_values(j)[1]
            if ncol == 3:
                bde =work_sheet.row_values(j)[2]
            else:
                bde = 0
            dict_pkas[smile] = pka
            dict_bdes[smile] = bde
    return dict_pkas, dict_bdes

def get_fps(smiles_s):
    fps = {}
    for smiles in smiles_s:
        for smile in smiles:
            mol = Chem.MolFromSmiles(smile)
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
            fps[smile] = fp
    return fps

def get_smiles(file_name):
    smiles_s = []
    exx = xlrd2.open_workbook(file_name)
    sheet_names = exx.sheet_names()
    num_sheets = len(sheet_names)
    for i in range(num_sheets):
        smiles = []
        work_sheet = exx.sheet_by_index(i)
        nrows = work_sheet.nrows
        ncol = work_sheet.ncols
        for j in range(1,nrows):
            smiles.append(work_sheet.row_values(j)[0])
        smiles_s.append(smiles)
    return smiles_s

def get_products(file_name):
    product_dict = {}
    product_smile = []
    exx_pro = xlrd2.open_workbook(file_name)
    work_sheet = exx_pro.sheet_by_index(0)
    nrows = work_sheet.nrows
    for i in range(1,nrows):
        chem1 = work_sheet.row_values(i)[0]
        chem2 = work_sheet.row_values(i)[1]
        product = work_sheet.row_values(i)[2]
        product_dict[(chem1,chem2)] = product
        product_smile.append(product)
    return product_dict, product_smile