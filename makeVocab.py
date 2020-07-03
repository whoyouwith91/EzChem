import os, sys, math, json, argparse, logging, time, random
from tokenizers import Tokenizer
from datetime import datetime
import torch
import numpy as np
import pandas as pd
from featurization import *
from sklearn.model_selection import train_test_split
from tokenizers.implementations import ByteLevelBPETokenizer

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--dataset', type=str, choices=['zinc'])
    parser.add_argument('--save_path', type=str)
    return parser.parse_args()


def main():
    args = parse_input_arguments()
    tokenizer = ByteLevelBPETokenizer()


    ### This is only for training a new text base in order to obtain a vocab file. 
    ### If you are done with this step, you should skip it. 
    tokenizer.train(files=args.dataset, vocab_size=52000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    ### save out the vocab file 
    tokenizer.save(args.save_path)

    

