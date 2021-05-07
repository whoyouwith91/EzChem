from typing import List, Tuple, Union
from rdkit import Chem

#__all__ = [get_atom_fdim, get_bond_fdim, MolGraph]

######################## Define atom features and bond features ##############
ATOM_FEATURES = {
    'atom_symbol': ['H', 'C', 'N', 'O', 'S', 'F', 'I', 'P', 'Cl', 'Br'],
    'atom_degree': [0, 1, 2, 3, 4, 5],
    'atom_explicitValence': [0, 1, 2, 3, 4, 5, 6],
    'atom_implicitValence': [0, 1, 2, 3, 4, 5],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# allowable node and edge features
BOND_FEATURES = {
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
        Chem.rdchem.BondDir.EITHERDOUBLE
    ]
}

######################## Define atom features and bond features ##############


def onek_encoding_unk(value, choices: List[int]) -> List[int]:
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding

def get_atom_fdim():
    return sum(len(choices)+1 for choices in ATOM_FEATURES.values()) + 1

def get_bond_fdim():
    return sum(len(choices)+1 for choices in BOND_FEATURES.values()) + 1

def get_atom_features(atom):
    features = onek_encoding_unk(atom.GetSymbol(), ATOM_FEATURES['atom_symbol']) + \
           onek_encoding_unk(atom.GetHybridization(), ATOM_FEATURES['hybridization']) + \
           onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['atom_degree']) + \
           onek_encoding_unk(atom.GetExplicitValence(), ATOM_FEATURES['atom_explicitValence']) + \
           onek_encoding_unk(atom.GetImplicitValence(), ATOM_FEATURES['atom_implicitValence']) + \
           [1 if atom.GetIsAromatic() else 0]
    return features


def bond_features_old(bond: Chem.rdchem.Bond) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for a bond.
    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    if bond is None:
        fbond = [1] + [0] * (BOND_FDIM - 1)
    else:
        bt = bond.GetBondType()
        fbond = [
            0,  # bond is not None
            bt == Chem.rdchem.BondType.SINGLE,
            bt == Chem.rdchem.BondType.DOUBLE,
            bt == Chem.rdchem.BondType.TRIPLE,
            bt == Chem.rdchem.BondType.AROMATIC,
            (bond.GetIsConjugated() if bt is not None else 0),
            (bond.IsInRing() if bt is not None else 0)
        ]
    return fbond

def get_bond_features(bond):
    fbond = [BOND_FEATURES['possible_bonds'].index(bond.GetBondType())] + 
            [BOND_FEATURES['possible_bond_dirs'].index(bond.GetBondDir())]
    return fbond

class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.
    A MolGraph computes the following attributes:
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, mol: Union[str, Chem.Mol], bfForModel='1-GNN'):
        """
        Computes the graph structure and featurization of a molecule.
        :param mol: A SMILES string or an RDKit molecule.
        """
        # Convert SMILES to RDKit molecule if necessary
        if type(mol) == str:
            mol = Chem.MolFromSmiles(mol)
        if type(mol) == Chem.rdchem.Mol:
            mol = mol

        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.real_f_bonds = []
        # Get atom features
        self.f_atoms = [get_atom_features(atom) for atom in mol.GetAtoms()]
        if bfForModel in ['1-GNN', '1-2-GNN', '1-2-efgs-GNN', '1-efgs-GNN', '1-interaction-GNN']:
           for bond in mol.GetBonds():
               bf = get_bond_features(bond)
               self.real_f_bonds.append(bf)
               self.real_f_bonds.append(bf)
           #self.real_f_bonds = [bond_features_new(bond) for bond in mol.GetBonds()]
        self.n_atoms = len(self.f_atoms)

        # Initialize atom to bond mapping for each atom
        for _ in range(self.n_atoms):
            self.a2b.append([])

        # Get bond features
        for a1 in range(self.n_atoms):
            for a2 in range(a1 + 1, self.n_atoms):
                bond = mol.GetBondBetweenAtoms(a1, a2)

                if bond is None:
                    continue

                f_bond = bond_features(bond)
                self.f_bonds.append(self.f_atoms[a1] + f_bond)
                self.f_bonds.append(self.f_atoms[a2] + f_bond)

                # Update index mappings
                b1 = self.n_bonds
                b2 = b1 + 1
                self.a2b[a2].append(b1)  # b1 = a1 --> a2
                self.b2a.append(a1)
                self.a2b[a1].append(b2)  # b2 = a2 --> a1
                self.b2a.append(a2)
                self.b2revb.append(b2)
                self.b2revb.append(b1)
                self.n_bonds += 2

        # Get bond index
        self.at_begin = []
        self.at_end = []
        for bond in mol.GetBonds():
            self.at_begin.append(bond.GetBeginAtom().GetIdx())
            self.at_begin.append(bond.GetEndAtom().GetIdx())
            self.at_end.append(bond.GetEndAtom().GetIdx())
            self.at_end.append(bond.GetBeginAtom().GetIdx())


