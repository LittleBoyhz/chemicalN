import numpy as np


def encode_dataset(sheet, char_to_idx, max_length, num_columns_of_sheet):
    num_rows = sheet.nrows

    yields = []
    encoding_of_dataset = []

    num_dim = 0
    for i in range(len(max_length)):
        num_dim += (max_length[i] * len(char_to_idx[i]))
        print("({}, {})".format(max_length[i], len(char_to_idx[i])))
    flattened_encoding_of_dataset = np.zeros((num_rows-1, num_dim))

    for row_idx in range(1, num_rows):
        row = sheet.row_values(row_idx)
        encoding_of_row = []
        flattened_encoding_of_row = np.empty((0,))
        yields.append(row[num_columns_of_sheet - 1])

        for j in range(num_columns_of_sheet - 1):
            smiles = row[j]
            encoding_of_smiles = smiles_to_one_hot(smiles, char_to_idx[j], max_length[j])
            encoding_of_row.append(encoding_of_smiles)
            flattened_encoding_of_row = np.concatenate((flattened_encoding_of_row, encoding_of_smiles.reshape(-1,)))

        encoding_of_dataset.append(encoding_of_row)
        flattened_encoding_of_dataset[row_idx-1, :] = flattened_encoding_of_row

    return encoding_of_dataset, flattened_encoding_of_dataset, np.array(yields)


def get_chars_and_max_length(sheet, num_columns_of_sheet):
    characters = list()
    max_length = list()
    for i in range(num_columns_of_sheet - 1):
        characters.append(set())
        max_length.append(0)

    num_rows = sheet.nrows
    for row_idx in range(1, num_rows):
        row = sheet.row_values(row_idx)

        for j in range(num_columns_of_sheet - 1):
            smiles = row[j]

            max_length[j] = max(max_length[j], len(smiles))

            for c in smiles:
                characters[j].add(c)

    char_to_idx = list()
    idx_to_char = list()
    for i in range(num_columns_of_sheet - 1):
        char_to_idx.append(dict((c, i) for i, c in enumerate(characters[i])))
        idx_to_char.append(dict((i, c) for i, c in enumerate(characters[i])))

    return characters, max_length, char_to_idx, idx_to_char


def one_hot_to_smiles(encoding, idx_to_char_all, max_length_all):
    num_dim = 0
    left = 0
    smiles_seq = []
    for i in range(len(max_length_all)):
        num_dim += (max_length_all[i] * len(idx_to_char_all[i]))

        one_hot = encoding[left:num_dim, ...].reshape(max_length_all[i], len(idx_to_char_all[i]))
        one_hot_max_idx = np.argmax(one_hot, axis=1)
        one_hot_max_value = np.amax(one_hot, axis=1)
        smiles = ""
        for k, j in enumerate(one_hot_max_idx):
            if one_hot_max_value[k] == 0:
                break
            smiles += idx_to_char_all[i][j]
        smiles_seq.append(smiles)

        left = num_dim

    return smiles_seq


def smiles_to_one_hot(smiles, char_to_idx, max_length):
    encoding = np.zeros((max_length, len(char_to_idx)))

    for i, c in enumerate(smiles):
        encoding[i, char_to_idx[c]] = 1

    return encoding
