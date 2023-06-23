import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def encode_dataset(sheet, num_columns_of_sheet, dim_molecule):
    num_rows = sheet.nrows
    num_dim = dim_molecule * (num_columns_of_sheet-1)
    flattened_encoding_of_dataset = np.zeros((num_rows-1, num_dim))
    yields = []

    for row_idx in range(1, num_rows):
        row = sheet.row_values(row_idx)
        yields.append(row[num_columns_of_sheet-1])

        for j in range(num_columns_of_sheet-1):
            smiles = row[j]
            mol = Chem.MolFromSmiles(smiles)
            encoding_of_smiles = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=dim_molecule))
            flattened_encoding_of_dataset[row_idx-1, j*dim_molecule:(j+1)*dim_molecule] = encoding_of_smiles

    return flattened_encoding_of_dataset, np.array(yields)
