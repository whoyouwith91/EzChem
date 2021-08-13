import pandas as pd
from rdkit import Chem

import sys
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.ML.Cluster import Butina

from rdkit.Chem.rdmolfiles import MolToXYZFile

def gen_conformers(mol, numConfs=300, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, enforceChirality=True):
    ids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=maxAttempts, pruneRmsThresh=pruneRmsThresh, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge, enforceChirality=enforceChirality, numThreads=0, randomSeed=1)
    return list(ids)

def calc_energy(mol, conformerId, minimizeIts):
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol), confId=conformerId)
    ff.Initialize()
    ff.CalcEnergy()
    results = {}
    if minimizeIts > 0:
        results["converged"] = ff.Minimize(maxIts=minimizeIts)
    results["energy_abs"] = ff.CalcEnergy()
    return results

def cluster_conformers(mol, mode="RMSD", threshold=2.0):
	if mode == "TFD":
		dmat = TorsionFingerprints.GetTFDMatrix(mol)
	else:
		dmat = AllChem.GetConformerRMSMatrix(mol, prealigned=False)
	rms_clusters = Butina.ClusterData(dmat, mol.GetNumConformers(), threshold, isDistData=True, reordering=True)
	return rms_clusters

def align_conformers(mol, clust_ids):
	rmslist = []
	AllChem.AlignMolConformers(mol, confIds=clust_ids, RMSlist=rmslist)
	return rmslist

def WriteLowestEnergyConformer(mol, conformerPropsDict, f_out):
    energy_abs = []
    for a in conformerPropsDict.values():
        energy_abs.append(a['energy_abs'])
    lowest_id = energy_abs.index(min(energy_abs))
    #w = Chem.SDWriter(f_out)
    #w.write(mol, confId=lowest_id)
    MolToXYZFile(mol, f_out, confId=lowest_id)


train = pd.read_csv('/scratch/dz1061/gcn/chemGraph/data/deepchem/logp/split/base/train.csv', names=['SMILES', 'target'])
valid = pd.read_csv('/scratch/dz1061/gcn/chemGraph/data/deepchem/logp/split/base/valid.csv',  names=['SMILES', 'target'])
test = pd.read_csv('/scratch/dz1061/gcn/chemGraph/data/deepchem/logp/split/base/test.csv',  names=['SMILES', 'target'])

all_ = pd.concat([train, valid, test])

for id_, smi in zip(range(1458, all_.shape[0]), all_[1458:]['SMILES']):
    try:
        #print(smi)
        mol = Chem.MolFromSmiles(smi)
        conformerPropsDict = {}
        m = Chem.AddHs(mol) # add hydrogen atoms 
        conformerIds = gen_conformers(m, 300, 1000, 0.1, True, True, True)
        for conformerId in conformerIds:
            # energy minimise (optional) and energy calculation
            props = calc_energy(m, conformerId, minimizeIts=200)
            conformerPropsDict[conformerId] = props
        WriteLowestEnergyConformer(m, conformerPropsDict, '/scratch/dz1061/gcn/chemGraph/data/deepchem/logp/split/base/MMFFXYZ/{}.xyz'.format(str(id_)))
    except:
        print(id_)
    #break