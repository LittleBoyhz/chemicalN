from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer, generate_fingerprints, RXNBERTMinhashFingerprintGenerator
)
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import pandas as pd

def canonicalize_with_dict(smi, can_smi_dict={}):
    if smi not in can_smi_dict.keys():
        return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    else:
        return can_smi_dict[smi]

def generate_buchwald_hartwig_rxns(sheet):
    fwd_template = '[F,Cl,Br,I]-[c;H0;D3;+0:1](:[c,n:2]):[c,n:3].[NH2;D1;+0:4]-[c:5]>>[c,n:2]:[c;H0;D3;+0:1](:[c,n:3])-[NH;D2;+0:4]-[c:5]'
    methylaniline = 'Cc1ccc(N)cc1'
    pd_catalyst = Chem.MolToSmiles(Chem.MolFromSmiles('O=S(=O)(O[Pd]1~[NH2]C2C=CC=CC=2C2C=CC=CC1=2)C(F)(F)F'))
    methylaniline_mol = Chem.MolFromSmiles(methylaniline)
    rxn = rdChemReactions.ReactionFromSmarts(fwd_template)
    products = []
    num_rows = sheet.nrows
    for row_idx in range(1,num_rows):
        row = sheet.row_values(row_idx)
        reacts = (Chem.MolFromSmiles(row[0]), methylaniline_mol)
        rxn_products = rxn.RunReactants(reacts)

        rxn_products_smiles = set([Chem.MolToSmiles(mol[0]) for mol in rxn_products])
        assert len(rxn_products_smiles) == 1
        products.append(list(rxn_products_smiles)[0])
    rxns = []
    can_smiles_dict = {}
    for row_idx in range(1,num_rows):
        row = sheet.row_values(row_idx)
        aryl_halide = canonicalize_with_dict(row[0], can_smiles_dict)
        can_smiles_dict[row[0]] = aryl_halide
        ligand = canonicalize_with_dict(row[3], can_smiles_dict)
        can_smiles_dict[row[3]] = ligand
        base = canonicalize_with_dict(row[2], can_smiles_dict)
        can_smiles_dict[row[2]] = base
        additive = canonicalize_with_dict(row[1], can_smiles_dict)
        can_smiles_dict[row[1]] = additive
        
        reactants = f"{aryl_halide}.{methylaniline}.{pd_catalyst}.{ligand}.{base}.{additive}"
        rxns.append(f"{reactants}>>{products[row_idx-1]}")
    return rxns

# 需按格式给出反应各组分
def encode_dataset(sheet, num_columns_of_sheet):
    num_rows = sheet.nrows
    
    converted_rxns_all = generate_buchwald_hartwig_rxns(sheet)
    
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    
    fps=[]
    #fps=rxnfp_generator.convert_batch(converted_rxns_all)
    for rxn in converted_rxns_all:
        fps.append(np.array(rxnfp_generator.convert(rxn)))
    fps=np.array(fps)
    
    yields = []
    for row_idx in range(1, num_rows):
        row = sheet.row_values(row_idx)
        yields.append(row[-1])

    return fps, np.array(yields)

def make_reaction_smiles(row):
    reactant_1_smiles = {
    '6-chloroquinoline': 'C1=C(Cl)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
    '6-Bromoquinoline': 'C1=C(Br)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
    '6-triflatequinoline': 'C1C2C(=NC=CC=2)C=CC=1OS(C(F)(F)F)(=O)=O.CCC1=CC(=CC=C1)CC',
    '6-Iodoquinoline': 'C1=C(I)C=CC2=NC=CC=C12.CCC1=CC(=CC=C1)CC', 
    '6-quinoline-boronic acid hydrochloride': 'C1C(B(O)O)=CC=C2N=CC=CC=12.Cl.O',
    'Potassium quinoline-6-trifluoroborate': '[B-](C1=CC2=C(C=C1)N=CC=C2)(F)(F)F.[K+].O',
    '6-Quinolineboronic acid pinacol ester': 'B1(OC(C(O1)(C)C)(C)C)C2=CC3=C(C=C2)N=CC=C3.O'
    }

    reactant_2_smiles = {
        '2a, Boronic Acid': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B(O)O', 
        '2b, Boronic Ester': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1B4OC(C)(C)C(C)(C)O4', 
        '2c, Trifluoroborate': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1[B-](F)(F)F.[K+]',
        '2d, Bromide': 'CC1=CC=C2C(C=NN2C3OCCCC3)=C1Br' 
    }

    catalyst_smiles = {
        'Pd(OAc)2': 'CC(=O)O~CC(=O)O~[Pd]'
    }

    ligand_smiles = {
        'P(tBu)3': 'CC(C)(C)P(C(C)(C)C)C(C)(C)C', 
        'P(Ph)3 ': 'c3c(P(c1ccccc1)c2ccccc2)cccc3', 
        'AmPhos': 'CC(C)(C)P(C1=CC=C(C=C1)N(C)C)C(C)(C)C', 
        'P(Cy)3': 'C1(CCCCC1)P(C2CCCCC2)C3CCCCC3', 
        'P(o-Tol)3': 'CC1=CC=CC=C1P(C2=CC=CC=C2C)C3=CC=CC=C3C',
        'CataCXium A': 'CCCCP(C12CC3CC(C1)CC(C3)C2)C45CC6CC(C4)CC(C6)C5', 
        'SPhos': 'COc1cccc(c1c2ccccc2P(C3CCCCC3)C4CCCCC4)OC', 
        'dtbpf': 'CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.CC(C)(C)P(C1=CC=C[CH]1)C(C)(C)C.[Fe]', 
        'XPhos': 'P(c2ccccc2c1c(cc(cc1C(C)C)C(C)C)C(C)C)(C3CCCCC3)C4CCCCC4', 
        'dppf': 'C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.C1=CC=C(C=C1)P([C-]2C=CC=C2)C3=CC=CC=C3.[Fe+2]', 
        'Xantphos': 'O6c1c(cccc1P(c2ccccc2)c3ccccc3)C(c7cccc(P(c4ccccc4)c5ccccc5)c67)(C)C',
        'None': ''
    }

    reagent_1_smiles = {
        'NaOH': '[OH-].[Na+]', 
        'NaHCO3': '[Na+].OC([O-])=O', 
        'CsF': '[F-].[Cs+]', 
        'K3PO4': '[K+].[K+].[K+].[O-]P([O-])([O-])=O', 
        'KOH': '[K+].[OH-]', 
        'LiOtBu': '[Li+].[O-]C(C)(C)C', 
        'Et3N': 'CCN(CC)CC', 
        'None': ''
    }

    solvent_1_smiles = {
        'MeCN': 'CC#N.O', 
        'THF': 'C1CCOC1.O', 
        'DMF': 'CN(C)C=O.O', 
        'MeOH': 'CO.O', 
        'MeOH/H2O_V2 9:1': 'CO.O', 
        'THF_V2': 'C1CCOC1.O'
    }
    precursors = f" {reactant_1_smiles[row['Reactant_1_Name']]}.{reactant_2_smiles[row['Reactant_2_Name']]}.{catalyst_smiles[row['Catalyst_1_Short_Hand']]}.{ligand_smiles[row['Ligand_Short_Hand']]}.{reagent_1_smiles[row['Reagent_1_Short_Hand']]}.{solvent_1_smiles[row['Solvent_1_Short_Hand']]} "
    product = 'C1=C(C2=C(C)C=CC3N(C4OCCCC4)N=CC2=3)C=CC2=NC=CC=C12'
    #print(precursors, product)
    can_precursors = Chem.MolToSmiles(Chem.MolFromSmiles(precursors.replace('...', '.').replace('..', '.').replace(' .', '').replace('. ', '').replace(' ', '')))
    can_product = Chem.MolToSmiles(Chem.MolFromSmiles(product))
    
    return f"{can_precursors}>>{can_product}"

def generate_SM():
    df = pd.read_excel('../datasets/real/real_5/real_5.xlsx',engine='openpyxl')
    rxns = [make_reaction_smiles(row) for i, row in df.iterrows()]
    yields = list(df['Product_Yield_PCT_Area_UV']/ 100. * 100.)
    
    model, tokenizer = get_default_model_and_tokenizer()
    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)
    
    fps=[]
    #fps=rxnfp_generator.convert_batch(converted_rxns_all)
    for rxn in rxns:
        fps.append(np.array(rxnfp_generator.convert(rxn)))
    fps=np.array(fps)

    return fps, np.array(yields)