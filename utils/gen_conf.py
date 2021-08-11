##https://gist.github.com/tdudgeon/b061dc67f9d879905b50118408c30aac

import sys
from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints
from rdkit.ML.Cluster import Butina


def gen_conformers(mol, numConfs=100, maxAttempts=1000, pruneRmsThresh=0.1, useExpTorsionAnglePrefs=True, useBasicKnowledge=True, enforceChirality=True):
	ids = AllChem.EmbedMultipleConfs(mol, numConfs=numConfs, maxAttempts=maxAttempts, pruneRmsThresh=pruneRmsThresh, useExpTorsionAnglePrefs=useExpTorsionAnglePrefs, useBasicKnowledge=useBasicKnowledge, enforceChirality=enforceChirality, numThreads=0, randomSeed=1)
	return list(ids)
	
def write_conformers_to_sdf(mol, filename, rmsClusters, conformerPropsDict, minEnergy):
	w = Chem.SDWriter(filename)
	for cluster in rmsClusters:
		for confId in cluster:
			for name in mol.GetPropNames():
				mol.ClearProp(name)
			conformerProps = conformerPropsDict[confId]
			mol.SetIntProp("conformer_id", confId + 1)
			for key in conformerProps.keys():
				mol.SetProp(key, str(conformerProps[key]))
			e = conformerProps["energy_abs"]
			if e:
				mol.SetDoubleProp("energy_delta", e - minEnergy)
			w.write(mol, confId=confId)
	w.flush()
	w.close()
	
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
		
	
if len(sys.argv) < 4:
	print "Usage: conf_gen.py <sdf input> <num conformers> <max attempts> <prune threshold> <cluster method: (RMSD|TFD) = RMSD> <cluster threshold = 0.2> <minimize iterations: = 0>"
	exit()
input_file = sys.argv[1]
numConfs = int(sys.argv[2])
maxAttempts = int(sys.argv[3])
pruneRmsThresh = float(sys.argv[4])
if len(sys.argv) > 5: clusterMethod = sys.argv[5]
else: clusterMethod = "RMSD"
if len(sys.argv) > 6: clusterThreshold = float(sys.argv[6])
else: clusterThreshold = 2.0
if len(sys.argv) > 7: minimizeIterations = int(sys.argv[7]) 
else: minimizeIterations = 0

suppl = Chem.ForwardSDMolSupplier(input_file)
i=0
for mol in suppl:
	i = i+1
	if mol is None: continue
	m = Chem.AddHs(mol)
	# generate the confomers
	conformerIds = gen_conformers(m, numConfs, maxAttempts, pruneRmsThresh, True, True, True)
	conformerPropsDict = {}
	for conformerId in conformerIds:
		# energy minimise (optional) and energy calculation
		props = calc_energy(m, conformerId, minimizeIterations)
		conformerPropsDict[conformerId] = props
	# cluster the conformers
	rmsClusters = cluster_conformers(m, clusterMethod, clusterThreshold)

	print "Molecule", i, ": generated", len(conformerIds), "conformers and", len(rmsClusters), "clusters"
	rmsClustersPerCluster = []
	clusterNumber = 0
	minEnergy = 9999999999999
	for cluster in rmsClusters:
		clusterNumber = clusterNumber+1
		rmsWithinCluster = align_conformers(m, cluster)
		for conformerId in cluster:
			e = props["energy_abs"]
			if e < minEnergy:
				minEnergy = e
			props = conformerPropsDict[conformerId]
			props["cluster_no"] = clusterNumber
			props["cluster_centroid"] = cluster[0] + 1
			idx = cluster.index(conformerId)
			if idx > 0:
				props["rms_to_centroid"] = rmsWithinCluster[idx-1]
			else:
				props["rms_to_centroid"] = 0.0

	write_conformers_to_sdf(m, str(i) + ".sdf", rmsClusters, conformerPropsDict, minEnergy)