import numpy as np
import os
import xlrd2

def get_dict(sheet_pka, dataset):
    pka_dict = {}
    bde_dict = {}
    pka_dict[""] = 0
    bde_dict[""] = 0
    if dataset == 4:
        for i in range(1,26):
            row_pka = sheet_pka.row_values(i)
            smiles = row_pka[0]
            value = row_pka[1]
            pka_dict[smiles] = value
        for i in range(27,42):
            row_bde = sheet_pka.row_values(i)
            smiles = row_bde[0]
            value = row_bde[1]
            bde_dict[smiles] = value
    elif dataset == 5:
        for i in range(1,34):
            row_pka = sheet_pka.row_values(i)
            smiles = row_pka[0]
            value = row_pka[1]
            pka_dict[smiles] = value
        for i in range(35,46):
            row_bde = sheet_pka.row_values(i)
            smiles = row_bde[0]
            value = row_bde[1]
            bde_dict[smiles] = value
    return pka_dict, bde_dict

def get_dict_6(excel):
    pka_dict = {}
    bde_dict = {}
    sheet_names = excel.sheet_names()
    num_sheets = len(sheet_names)
    for i in range(num_sheets):
        work_sheet = excel.sheet_by_index(i)
        nrows = work_sheet.nrows
        ncol = work_sheet.ncols
        for j in range(1,nrows):
            smile = work_sheet.row_values(j)[0]
            pka = work_sheet.row_values(j)[1]
            if ncol == 3:
                bde =work_sheet.row_values(j)[2]
            else:
                bde = 0
            pka_dict[smile] = pka
            bde_dict[smile] = bde
    return pka_dict, bde_dict

def encode_dataset(sheet, num_columns_of_sheet, args, norm=False, dataset=4):
    num_rows = sheet.nrows
    flattened_encoding_of_dataset = []
    yields = []
    if norm:
        excel_path = os.path.join(args.data_folder, args.dataset, "pka_bde_01.xlsx")
    else:
        excel_path = os.path.join(args.data_folder, args.dataset, "pka_bde.xlsx")
    excel = xlrd2.open_workbook(excel_path)
    if dataset != 6:
        sheet_pka = excel.sheet_by_name(args.sheet_name)
        pka_dict, bde_dict = get_dict(sheet_pka, dataset)
    else:
        pka_dict, bde_dict = get_dict_6(excel)

    for row_idx in range(1, num_rows):  # idx: 1-(num_rows-1) row: 2nd->last row
        row = sheet.row_values(row_idx)
        yields.append(row[num_columns_of_sheet - 1])

        values = []
        
        for j in range(num_columns_of_sheet - 1):  # j: 0-(num_column-2) column: 1st->second last row
            smiles = row[j]
            pka_value = 0
            bde_value = 0
            if smiles in pka_dict:
                pka_value = pka_dict[smiles]
            if smiles in bde_dict:
                bde_value = bde_dict[smiles]
            values.append(pka_value)
            values.append(bde_value)

        flattened_encoding_of_dataset.append(np.array(values))

    return np.array(flattened_encoding_of_dataset), np.array(yields)