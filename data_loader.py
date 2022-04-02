import pandas as pd
import numpy as np
import torch
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
from torch.utils.data import Dataset, DataLoader, BatchSampler, Sampler

tokenizer = DistilBertTokenizer(f'{MODEL_PATH}vocab.txt')
config = DistilBertConfig.from_json_file(f'{MODEL_PATH}config.json')
distilbert = DistilBertModel(config)
state_dict = torch.load(f'{MODEL_PATH}model.bin')
distilbert.load_state_dict(state_dict)


class TweetDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.X = df['text'].values
        
        if TARGET in df:
            self.Y = df['selected_text'].values
        else:
            self.Y = None
        
        self.tokenizer = tokenizer
        
    def __getitem__(self, idx):
        
        x = self.X[idx]
        x = self.tokenizer.encode_plus(x, max_length=512, add_special_tokens=True, return_tensors='pt')
                                       
        attn_mask = x['attention_mask'].squeeze()

        x = x['input_ids'].squeeze()
        
        
        if self.Y is not None:
            
            y = self.Y[idx]
            y = self.tokenizer.encode_plus(y, max_length=512, return_tensors='pt', add_special_tokens=True, return_special_tokens_mask=True)
            mask = y['special_tokens_mask']
           
            y = y['input_ids'].squeeze()

            y = y[1:-1] 
        
            y_new = torch.zeros_like(x)
            
            idxs = torch.where(x==y[0])[0]
  
            for idx in idxs:
                x1 = x[idx: idx + len(y)]          
                if torch.all(x1.eq(y)):
                    y_new[idx: idx + len(y)] = 1
       
                    break

            assert y_new.sum().item()==len(y)
            
            return x, attn_mask, y_new.float()
        else:
            return x, attn_mask
            
    def __len__(self):
        return self.X.shape[0]


class BatchRandomSampler(Sampler):
    def __init__(self, data_source, batch_size):
        self.data_source = data_source
        self.batch_size = batch_size

    def __iter__(self):
        n_of_batches = len(self.data_source)//batch_size
        batches = np.arange(batch_size*n_of_batches).reshape((n_of_batches, batch_size))
        batches = np.random.permutation(batches)

        return iter(batches)

    def __len__(self):
        return len(self.data_source)//batch_size
    
def custom_collate(batch):
    
    if len(batch[0])==3:
        x, attn_mask, y = zip(*batch)
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=0)
    elif len(batch[0])==2:
        x, attn_mask = zip(*batch)
        y = None
        
    x = torch.nn.utils.rnn.pad_sequence(x, batch_first=True, padding_value=tokenizer.pad_token_id)

    attn_mask = torch.nn.utils.rnn.pad_sequence(attn_mask, batch_first=True, padding_value=0)

    return x, attn_mask, y