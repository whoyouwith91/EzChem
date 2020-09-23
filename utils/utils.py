ALLOWABLE_ATOM_SYMBOLS = ['H', 'C', 'N', 'O', 'S', 'F', 'I', 'P', 'Cl', 'Br']

# filter out molecules that are within elements groups
with open('/beegfs/dz1061/gcn/chemGraph/data/smiles/atom10_smiles.txt', 'w') as f:
    for i in text:
        mol = Chem.MolFromSmiles(i)
        elements = list(Counter(atom.GetSymbol() for atom in mol.GetAtoms()).keys())
        if set(elements) < set(ALLOWABLE_ATOM_SYMBOLS):
            f.write(i)
            f.write('\n')
