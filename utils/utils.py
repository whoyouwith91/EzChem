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


def get_hbond_donor_indice(m):
    """
    indice = m.GetSubstructMatches(HDonorSmarts)
    if len(indice)==0: return np.array([])
    indice = np.array([i for i in indice])[:,0]
    return indice
    """
    # smarts = ["[!$([#6,H0,-,-2,-3])]", "[!H0;#7,#8,#9]"]
    smarts = ["[!#6;!H0]"]
    indice = []
    for s in smarts:
        s = Chem.MolFromSmarts(s)
        indice += [i[0] for i in m.GetSubstructMatches(s)]
    indice = np.array(indice)
    return indice


def get_hbond_acceptor_indice(m):
    # smarts = ["[!$([#6,F,Cl,Br,I,o,s,nX3,#7v5,#15v5,#16v4,#16v6,*+1,*+2,*+3])]",
    #          "[#6,#7;R0]=[#8]"]
    smarts = [
        '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +
                                     '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +
                                     '$([nH0,o,s;+0])]']
    indice = []
    for s in smarts:
        #print(s)
        s = Chem.MolFromSmarts(s)
        indice += [i[0] for i in m.GetSubstructMatches(s)]
    indice = np.array(indice)
    return indice
