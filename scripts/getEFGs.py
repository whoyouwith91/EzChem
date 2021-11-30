from three_level_frag import cleavage, AtomListToSubMol, standize, mol2frag, WordNotFoundError, counter
from ifg import identify_functional_groups
from three_level_frag import cleavage

import pandas as pd
from collections import Counter
from rdkit import Chem
import glob, sys
import torch 

def getVocab(df):
    word_ours = Counter()
    fail_smiles = []
    for idx, smiles in enumerate(df['SMILES']):
        try:
            mol = Chem.MolFromSmiles(smiles)
            a,b = mol2frag(mol)
            word_ours.update(a+b)
        except:
            fail_smiles.append(smiles)

        if idx % 10000 == 0:
            print(idx)
            sys.stdout.flush()
    return word_ours, fail_smiles

for file in glob.glob('/scratch/dz1061/gcn/chemGraph/data/smiles/PubChem/train*'):
    print(file)
    sys.stdout.flush()
    df = pd.read_csv(file, names=['SMILES'])
    vocab, fail_smi = getVocab(df)

    torch.save(vocab, file.split('.')[0] + '_vocab.pt')
    print(len(fail_smi), df.shape[0])
    sys.stdout.flush()
