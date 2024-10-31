import pandas as pd
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity


# function to canonicalize SMILES
def normalize_smiles(smi, canonical=True, isomeric=False):
    try:
        normalized = Chem.MolToSmiles(
            Chem.MolFromSmiles(smi), canonical=canonical, isomericSmiles=isomeric
        )
    except:
        normalized = None
    return normalized

# function to calculate pairwise Tanimoto similarity
def calculate_tanimoto_similarities(fps1, fps2):
    similarities = []
    for i in range(len(fps1)):
            sim = TanimotoSimilarity(fps1[i], fps2[i])
            similarities.append(sim)
    return similarities

def canon_smiles(smiles):
    new_smiles = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        s = Chem.MolToSmiles(mol)
        new_smiles.append(s)
    return new_smiles

def load_cwa_smiles(remove_chirality=True, remove_isomer=True):
    df = pd.read_excel('data/CWA-DB.xlsx')
    smiles = list(df['SMILES'])
    if remove_chirality:
        smiles = [s.replace('@','') for s in smiles]
    df = pd.read_excel('data/CWA_SMILES-Williams.xlsx')
    cxsmiles = list(df['Smiles (CXSmiles)'])
    for cs in cxsmiles:
        mol = Chem.MolFromSmiles(cs)
        if mol is not None:
            if remove_isomer:
                s = Chem.MolToSmiles(mol).replace('/','').replace('\\','')
            else:
                s = Chem.MolToSmiles(mol)
            if '.' in s:
                s_split = s.split('.')
                for sp in s_split:
                    smiles.append(sp)
            else:
                smiles.append(s)
    smiles = canon_smiles(smiles)
    smiles = list(set(smiles))
    return smiles
