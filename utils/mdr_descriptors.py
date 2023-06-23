from mordred import Calculator, descriptors
import numpy as np
from rdkit import Chem
from sklearn.preprocessing import MinMaxScaler

def encode_dataset(sheet, num_columns_of_sheet):
    calc = Calculator(descriptors, ignore_3D=True)
    dim_molecule = len(Calculator(descriptors, ignore_3D=True).descriptors)
    num_rows = sheet.nrows
    num_dim = dim_molecule * (num_columns_of_sheet - 1)
    flattened_encoding_of_dataset = np.zeros((num_rows - 1, num_dim))
    yields = []
    
    def get_dict():
        sss = set()
        dict = {}
        for row_idx in range(1, num_rows):  # idx: 1-(num_rows-1) row: 2nd->last row
            row = sheet.row_values(row_idx)
            for j in range(num_columns_of_sheet - 1):  # j: 0-(num_column-2) column: 1st->second last row
                smiles = row[j]
                sss.add(smiles)
        for smile in sss:
            if smile == '':
                encoding_of_smiles = np.zeros(1613)
            else:
                mol = Chem.MolFromSmiles(smile)
                encoding_of_smiles = np.array(calc(mol))
            encoding_of_smiles = np.nan_to_num(encoding_of_smiles)
            dict[smile] = encoding_of_smiles
        return dict
    
    dict = get_dict()

    for row_idx in range(1, num_rows):  # idx: 1-(num_rows-1) row: 2nd->last row
        row = sheet.row_values(row_idx)
        yields.append(row[num_columns_of_sheet - 1])

        for j in range(num_columns_of_sheet - 1):  # j: 0-(num_column-2) column: 1st->second last row
            smiles = row[j]
            encoding_of_smiles = dict[smiles]
            float_arr = encoding_of_smiles.astype(np.float64)
            where_are_nan = np.isnan(float_arr)
            encoding_of_smiles[where_are_nan] = 0  # in case there exists nan among Mordred descriptors
            flattened_encoding_of_dataset[row_idx - 1, j * dim_molecule:(j + 1) * dim_molecule] = encoding_of_smiles

    scaler = MinMaxScaler()
    flattened_encoding_of_dataset = scaler.fit_transform(flattened_encoding_of_dataset)
    return flattened_encoding_of_dataset, np.array(yields)
