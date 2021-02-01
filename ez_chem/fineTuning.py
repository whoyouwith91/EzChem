import os, sys, math, json, argparse, logging, time, random
import pandas as pd
import numpy as np

import torch
from torchtext.vocab import Vocab
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from transformers import LineByLineTextDataset
from transformers import RobertaForMaskedLM

from helper import *
from tokenization import *
from tokenizers import ByteLevelBPETokenizer
from transformers import RobertaTokenizerFast
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import DataCollatorForLanguageModeling
from transformers import RobertaForMaskedLM
from transformers import Trainer, TrainingArguments
from transformers import AdamW, get_linear_schedule_with_warmup

def parse_input_arguments():
    parser = argparse.ArgumentParser(description='NLP configurations')
    '''
    The followings are very basic parameters. 
    '''
    parser.add_argument('--dataset', type=str,
                        help='dataset for training')
    parser.add_argument('--tokenizer', type=str,
                        help='method to generate tokens and ids')
    parser.add_argument('--vocabPath', type=str,
                        help='directory to save the vocab files')
    parser.add_argument('--allDataPath', type=str, default='/beegfs/dz1061/gcn/chemGraph/data')
    parser.add_argument('--running_path', type=str,
                        help='path to save model', default='/beegfs/dz1061/gcn/chemGraph/results')
    parser.add_argument('--trainingStyle', type=str,
                        help='pretraining or fine-tuning')
    parser.add_argument('--uncertainty',  type=str, default='None')
    parser.add_argument('--uncertainty_method',  type=str, default='None')
    parser.add_argument('--taskType', type=str, default='None')
    parser.add_argument('--D', type=int, default=768) # hidden size
    parser.add_argument('--H', type=int, default=12)  #               
    parser.add_argument('--L', type=int, default=12)
    parser.add_argument('--maxLength', type=int, default=150)
        
    parser.add_argument('--preTrainedPath', type=str) 
    parser.add_argument('--Epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--loss', type=str, choices=['l1', 'l2', 'smooth_l1', 'dropout', 'vae'])
    parser.add_argument('--lr', type=float, default=0.00005)
    parser.add_argument('--style', type=str)
    parser.add_argument('--pooling', type=str, choices=['add', 'mean', 'max', 'cat', 'set2set'], default='add')
    return parser.parse_args()

class WSDataset(Dataset):

    def __init__(self, smiles, targets, tokenizer, max_len):
        self.smi = smiles
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
  
    def __len__(self):
        return len(self.smi)
  
    def __getitem__(self, item):
        smis = str(self.smi[item])
        target = self.targets[item]

        encoding = self.tokenizer.encode_plus(
                                                smis,
                                                add_special_tokens=True,
                                                max_length=self.max_len,
                                                return_token_type_ids=False,
                                                pad_to_max_length=True,
                                                return_attention_mask=True,
                                                return_tensors='pt',
                                                truncation=True
                                                )

        return {
                'smiles_text': smis,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(target, dtype=torch.float)
                }

def create_data_loader(df, tokenizer, max_len, batch_size):
    ds = WSDataset(
                    smiles=df.SMILES.to_numpy(),
                    targets=df.target.to_numpy(),
                    tokenizer=tokenizer,
                    max_len=max_len
                )

    return DataLoader(
                        ds,
                        batch_size=batch_size,
                        num_workers=4
                    )

class robertaProperty(nn.Module):

    def __init__(self, config):
        super(robertaProperty, self).__init__()
        self.config = config 
        self.bert = RobertaForMaskedLM.from_pretrained(config['preTrainedPath']) # todo 
        self.l1 = nn.Sequential(nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size//2), nn.ReLU())
        if self.config['pooling'] == 'cat':
           self.l1 = nn.Sequential(nn.Linear(self.bert.config.hidden_size*3, self.bert.config.hidden_size//2), nn.ReLU())
        self.l2 = nn.Sequential(nn.Linear(self.bert.config.hidden_size//2, self.bert.config.hidden_size//4), nn.ReLU())
        self.l3 = nn.Sequential(nn.Linear(self.bert.config.hidden_size//4, 1))
    
  
    def forward(self, input_ids, attention_mask):
        output = self.bert(
        input_ids=input_ids,
        attention_mask=attention_mask
        )
        
        if self.config['pooling'] == 'add':
           pooled_output = torch.sum(output[-1][-1], dim=1) # last output of decoding layers, mean pooling
        if self.config['pooling'] == 'mean':
           pooled_output = torch.mean(output[-1][-1], dim=1) # last output of decoding layers, mean pooling
        if self.config['pooling'] == 'cat':
           pooled_output = torch.cat((torch.mean(output[-1][-1], dim=1), torch.max(output[-1][-1], dim=1).values, output[-1][-1][:,0,:]), dim=1)
        output = self.l1(pooled_output)
        output = self.l2(output)
        output = self.l3(output)
        #output = self.drop(pooled_output)
        return output.squeeze()

def train_epoch(model, data_loader, optimizer, scheduler, device, config):
    model.train()

    preds, labels = [], []
    for d in data_loader:
        optimizer.zero_grad()
        
        input_ids_ = d["input_ids"].to(device)
        attention_mask_ = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(
              input_ids=input_ids_,
              attention_mask=attention_mask_
            )
        
        #print(outputs.squeeze().shape, targets.shape)
        loss = get_loss_fn(config['loss'])(targets, outputs)
        #print(loss)
        #print(targets, outputs.squeeze())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        #print(loss)
        preds.extend(outputs.cpu().data.numpy())
        labels.extend(targets.cpu().data.numpy())

    return np.mean((np.array(preds).reshape(-1,) - np.array(labels))**2)**0.5

def eval_epoch(model, data_loader, optimizer, device, config):
    model.eval()

    preds, labels = [], []
    with torch.no_grad():
        for d in data_loader:
            input_ids_ = d["input_ids"].to(device)
            attention_mask_ = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(
                input_ids=input_ids_,
                attention_mask=attention_mask_
            )
        
        preds.extend(outputs.cpu().data.numpy())
        labels.extend(targets.cpu().data.numpy())

    return np.mean((np.array(preds).reshape(-1,) - np.array(labels))**2)**0.5

def main():
    args = parse_input_arguments()
    this_dic = vars(args)
    this_dic['model'] = 'Roberta'
    this_dic['running_path'] = args.running_path = os.path.join(this_dic['running_path'], this_dic['dataset'], this_dic['model'], this_dic['tokenizer'], this_dic['style'])

    if not os.path.exists(os.path.join(args.running_path, 'trained_model/')):
        os.makedirs(os.path.join(args.running_path, 'trained_model/'))
    if not os.path.exists(os.path.join(args.running_path, 'best_model/')):
        os.makedirs(os.path.join(args.running_path, 'best_model/'))
    createResultsFile(this_dic, name='data.txt')

    if this_dic['tokenizer'] == 'BPE':
        tokenizer = RobertaTokenizerFast.from_pretrained(this_dic['vocabPath'], max_len=512)
    if this_dic['tokenizer'] == 'MOL':
        tokenizer = SmilesTokenizer(vocab_file=os.path.join(this_dic['vocabPath'], 'vocab.txt'))
    vocab_size = tokenizer.vocab_size
    this_dic['vocab_size'] = vocab_size
    max_len = 150
    #ws
    train = pd.read_csv(os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'split', 'base', 'train.csv'), names=['SMILES', 'target']) # todo
    valid = pd.read_csv(os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'split', 'base', 'valid.csv'), names=['SMILES', 'target']) # todo 
    test = pd.read_csv(os.path.join(this_dic['allDataPath'], this_dic['dataset'], 'split', 'base', 'test.csv'), names=['SMILES', 'target']) # todo


    train_data_loader = create_data_loader(train, tokenizer, this_dic['maxLength'], this_dic['batch_size'])
    val_data_loader = create_data_loader(valid, tokenizer, this_dic['maxLength'], this_dic['batch_size'])
    test_data_loader = create_data_loader(test, tokenizer, this_dic['maxLength'], this_dic['batch_size'])
    
    device = torch.device('cuda')
    model = robertaProperty(this_dic).to(device)

    optimizer = AdamW(model.parameters(), lr=this_dic['lr'], eps=1e-8)
    total_steps = len(train_data_loader) * 4
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0, # Default value
                                                num_training_steps=total_steps)
    this_dic.update(model.config)
    saveConfig(this_dic, name='config.json')
    best_val_error = float("inf")
    for epoch in range(this_dic['Epochs']):
        saveContents = []

        time_tic = time.time()
        train_error = train_epoch(model, train_data_loader, optimizer, scheduler, device, this_dic)
        time_toc = time.time()
        val_error = eval_epoch(model, val_data_loader, optimizer, device, this_dic)
        test_error = eval_epoch(model, test_data_loader, optimizer, device, this_dic)

        saveContents.append([model, epoch, time_toc, time_tic, train_error,  \
                        val_error, test_error, param_norm(model), grad_norm(model)])
        saveToResultsFile(this_dic, saveContents[0], name='data.txt')
        
        if best_val_error > np.sum(val_error):
            best_val_error = np.sum(val_error)
            #logging.info('Saving models...')
            torch.save(model.state_dict(), os.path.join(this_dic['running_path'], 'best_model', 'model_'+str(epoch)+'.pt'))

if __name__ == '__main__':
    main()
