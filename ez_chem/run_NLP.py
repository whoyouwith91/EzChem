import os, sys, math, json, argparse, logging, time, random, glob
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torchtext.vocab import Vocab
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import LineByLineTextDataset
from transformers import RobertaForMaskedLM

from tokenization import *
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
import transformers

transformers.logging.set_verbosity_info()
def parse_input_arguments():
    parser = argparse.ArgumentParser(description='NLP configurations')
    '''
    The followings are very basic parameters. 
    '''
    parser.add_argument('--dataset', type=str,
                        help='dataset for training')
    parser.add_argument('--tokenizer', type=str,
                        help='method to generate tokens and ids')
    parser.add_argument('--multiDatasets', action='store_true')
    parser.add_argument('--allDataPath', type=str, default='/beegfs/dz1061/gcn/chemGraph/data')
    parser.add_argument('--runningPath', type=str,
                        help='path to save model', default='/beegfs/dz1061/gcn/chemGraph/results')
    parser.add_argument('--trainingStype', type=str,
                        help='pretraining or fine-tuning')

    parser.add_argument('--D', type=int, default=768)
    parser.add_argument('--H', type=int, default=12)                
    parser.add_argument('--L', type=int, default=12)
    
    parser.add_argument('--Epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--saveSteps', type=int, default=500)
    parser.add_argument('--saveLimit', type=int, default=10)
    parser.add_argument('--logSteps', type=int, default=25)
    parser.add_argument('--evalSteps', type=int, default=25)
    parser.add_argument('--style', type=str)
    return parser.parse_args()

def main():
    args = parse_input_arguments()
    this_dic = vars(args)

    if this_dic['tokenizer'] == 'BPE':
        tokenizer = RobertaTokenizerFast.from_pretrained(os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'BPE'), max_len=512)
    if this_dic['tokenizer'] == 'MOL':
        tokenizer = SmilesTokenizer(vocab_file=os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'MOL', 'vocab.txt'))
    vocab_size = tokenizer.vocab_size
    this_dic['vocab_size'] = vocab_size
    
    if not this_dic['multiDatasets']:
        if this_dic['trainingStype'] in ['preTraining', 'fineTuning']: 
            train = LineByLineTextDataset(
                                tokenizer=tokenizer,
                                file_path=os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'train.txt'),
                                block_size=128)
    
            valid = LineByLineTextDataset(
                                tokenizer=tokenizer,
                                file_path=os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'valid.txt'),
                                block_size=128)
        if this_dic['trainingStype'] in ['fineTuning']: 
            test = LineByLineTextDataset(
                                tokenizer=tokenizer,
                                file_path=os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'train.txt'),
                                block_size=128)

    data_collator = DataCollatorForLanguageModeling(
                    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    
    #https://huggingface.co/transformers/model_doc/bert.html#transformers.BertConfig
    config = RobertaConfig(
                vocab_size=this_dic['vocab_size'],
                hidden_size=this_dic['D'], # dimension
                num_attention_heads=this_dic['H'], # number of attention heads
                num_hidden_layers=this_dic['L'], # number of layers
                type_vocab_size=1,
                output_hidden_states=True)
    
    model = RobertaForMaskedLM(config=config)
    this_dic['NumParams'] = model.num_parameters()
    this_dic['runningPath'] = os.path.join(this_dic['runningPath'], this_dic['dataset'], this_dic['tokenizer'], this_dic['style'])
    if this_dic['multiDatasets']:
        this_dic['Epochs'] = 1
    training_args = TrainingArguments(
                    output_dir=this_dic['runningPath'],  # todo
                    overwrite_output_dir=True,
                    num_train_epochs=this_dic['Epochs'],
                    do_train=True,
                    do_eval=True,
                    evaluate_during_training=True,
                    per_device_train_batch_size=this_dic['batch_size'],
                    save_steps=this_dic['saveSteps'],
                    save_total_limit=this_dic['saveLimit'],
                    logging_dir=os.path.join(this_dic['runningPath'], 'log'),
                    logging_steps=this_dic['logSteps'],
                    prediction_loss_only=True,
                    eval_steps=this_dic['evalSteps'])

    #saveConfig(this_dic, name='config.json')
    if not this_dic['multiDatasets']:
        trainer = Trainer(
                    model=model,
                    args=training_args,
                    data_collator=data_collator,
                    train_dataset=train,
                    eval_dataset=valid,
                    prediction_loss_only=True)
    
        trainer.train()
    else:
        numTrainFiles = len(glob.glob(os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'train*')))
        if this_dic['trainingStype'] in ['preTraining', 'fineTuning']: 
            for i in tqdm(range(72, numTrainFiles)):
                #print('Loading file: ' + os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'train_{}.txt'.format(str(i))))
                train = LineByLineTextDataset(
                                    tokenizer=tokenizer,
                                    file_path=os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'train_{}.txt'.format(str(i))),
                                    block_size=128)
                valid = LineByLineTextDataset(
                                    tokenizer=tokenizer,
                                    file_path=os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'valid.txt'),
                                    block_size=128)

                training_args.output_dir = os.path.join(this_dic['runningPath'], str(i))
                training_args.logging_dir = os.path.join(this_dic['runningPath'], str(i), 'log') 
                if i == 0:
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=train,
                        eval_dataset=valid,
                        prediction_loss_only=True)
                
                    trainer.train()
                else:
                    lastModel = sorted(glob.glob(os.path.join(this_dic['runningPath'], str(i-1), 'checkpoint*')), key=os.path.getmtime)[-1]
                    model.from_pretrained(lastModel)
                    trainer = Trainer(
                        model=model,
                        args=training_args,
                        data_collator=data_collator,
                        train_dataset=train,
                        eval_dataset=valid,
                        prediction_loss_only=True)
                    trainer.train()

    #https://huggingface.co/transformers/main_classes/model.html
    trainer.save_model(os.path.join(this_dic['runningPath'], 'Roberta'))

if __name__ == '__main__':
    main()

