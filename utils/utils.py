ALLOWABLE_ATOM_SYMBOLS = ['H', 'C', 'N', 'O', 'S', 'F', 'I', 'P', 'Cl', 'Br']

# filter out molecules that are within elements groups
with open('/beegfs/dz1061/gcn/chemGraph/data/smiles/atom10_smiles.txt', 'w') as f:
    for i in text:
        mol = Chem.MolFromSmiles(i)
        elements = list(Counter(atom.GetSymbol() for atom in mol.GetAtoms()).keys())
        if set(elements) < set(ALLOWABLE_ATOM_SYMBOLS):
            f.write(i)
            f.write('\n')
            
            from collections import Counter

# create a vocabunary
counter = Counter()

with open('/beegfs/dz1061/gcn/chemGraph/data/smiles/atom10_smiles.txt') as rf:
    for line in rf:
        items = tokenizer.tokenize(line.strip())
        counter.update(items)
        #items = self._tokenize(line.strip())
vocab = torchtext.vocab.Vocab(counter)

torch.save(vocab, '/beegfs/dz1061/gcn/chemGraph/data/smiles/atom10_vocab.pt')
