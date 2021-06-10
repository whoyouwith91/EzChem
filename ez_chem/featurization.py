from typing import List, Tuple, Union
from rdkit import Chem
#from descriptastorus.descriptors import rdDescriptors, rdNormalizedDescriptors

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

'''
Periodic properties: (in order as)
Element period #, element group #, atomic weight, VDW atomic radius, 
covalent atomic radius, ionization energy, electronegativity, 
election affinity
'''
ATOM_FEATURES_PERIODIC = {

    'H': [-1.460954261,-2.88512737,-0.8978528170000001,-1.953788528,-1.809863616,0.535333521,-1.020768417,-0.791439307],
    'C': [-0.568148879,-0.07850686700000001,-0.601798994,0.112718569,-0.357866064,-0.396995828,-0.425320174,-0.41149686700000004],
    'B': [-0.568148879,-0.294400752,-0.6341874829999999,1.021981691,-0.099733165,-1.578159015,-1.292973329,-1.147495385],
    'N': [-0.568148879,0.137387018,-0.548064941,-0.50723356,-0.519199125,0.908584278,0.40830736700000003,-1.3542617030000001],
    'O': [-0.568148879,0.353280902,-0.494444716,-0.6312239860000001,-0.680532186,0.543308965,1.088819645,-0.264446453],
    'S': [0.324656502,0.353280902,-0.06227609,0.5260199879999999,0.577865693,-0.755890787,-0.374281753,0.19611245800000002],
    'F': [-0.568148879,0.569174787,-0.413743398,-1.333836399,-0.970931697,2.060637097,2.007511221,1.138130846],
    'Cl': [0.324656502,0.569174787,0.02886699,0.319369279,0.481065856,0.28410705,0.61246105,1.345643613],
    'Br': [1.217461884,0.569174787,1.225136739,0.650010414,1.061864877,-0.176076042,0.272204911,1.157538515],
    'P': [0.324656502,0.137387018,-0.091500001,0.5260199879999999,0.6423989170000001,-0.70524672,-1.0377812240000002,-0.797410897],
    'I': [2.110267265,0.569174787,2.48986471,1.2699625429999999,1.674930511,-0.719602519,-0.23817929699999998,0.929125181]
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

def rdkit_2d_normalized_features_generator(mol):
        """
        Generates RDKit 2D normalized features for a molecule.
        :param mol: A molecule (i.e., either a SMILES or an RDKit molecule).
        :return: A 1D numpy array containing the RDKit 2D normalized features.
        """
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True) if type(mol) != str else mol
        generator = rdNormalizedDescriptors.RDKit2DNormalized()
        features = generator.process(smiles)[1:]
        return features
######################## Define atom features and bond features ##############

##############################################################################
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
    fbond = [BOND_FEATURES['possible_bonds'].index(bond.GetBondType())] + \
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

    def __init__(self, mol, periodic_features=False, bfForModel='1-GNN'):
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

        # Get atom/bond features
        if not periodic_features:
            self.f_atoms = [get_atom_features(atom) for atom in mol.GetAtoms()]
        else:
            self.f_atoms = [ATOM_FEATURES_PERIODIC[atom.GetSymbol()] for atom in mol.GetAtoms()]
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

                f_bond = get_bond_features(bond)
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


