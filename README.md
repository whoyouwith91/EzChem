EzChem
=====
## EzChem is a python package based on PyTorch. It can be used to solve problems in Chemistry, such as prediction of chemical reactions and physicochemical properties. 

## Installation
These packages need to be installed:
- [PyTorch](https://pytorch.org/)
```
# on Linux
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```
- [GPyTorch](https://gpytorch.ai/)
```
$ pip install gpytorch
```
- [PyTorch geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
- [k_gnn](https://github.com/chrsmrrs/k-gnn)
- [TorchText](https://pytorch.org/text/)
```
$ pip install torchtext
```
- [:)Transformers](https://github.com/huggingface/transformers#installation)
```
$ pip install transformers
```
- [RDKit](https://www.rdkit.org/docs/GettingStartedInPython.html)
```
$ conda install -c rdkit rdkit
```
- [Hyperopt](https://github.com/hyperopt/hyperopt)
```
$ pip install hyperopt
```
- [chemprop](https://github.com/chemprop/chemprop#option-1-conda)
- [torchcontrib](https://pypi.org/project/torchcontrib/)
```
pip install torchcontrib
```

## Data 
- Why and how to prepare data files? 
> Because it usually takes long time to process molecule datasets by reading the SMILES/InChI and generating corresponding molecular graphs. If the data processing and model training happen at same stage, training speed will slow down a lot, which is unnecessary. 
> Before prepraring the data files, you should: 1. Get ready of the train/valid/test set in 'csv' format. First column stores SMILES, second column stores target values. 
