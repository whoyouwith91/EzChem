import os, sys, math, json, argparse, logging, time, random
from tokenizers import Tokenizer
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from featurization import *
from sklearn.model_selection import train_test_split
from tokenizers.implementations import MolTokenizer

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--data_path', type=str, choices=['zinc'])
    parser.add_argument('--dataset', type=str, choices=['zinc'])
    parser.add_argument('--save_path', type=str)
    return parser.parse_args()


def main():
    args = parse_input_arguments()
    tokenizer = MolTokenizer(source_files=os.path.join(args.data_path, args.dataset)) # this will help find all unique tokens. 
    
    ### save out the vocab file 
    tokenizer.save_vocabulary(args.save_path) # currently, the vocab name is fixed in the moltokenizer script. you may need to change it.

    

