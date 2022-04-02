import numpy as np
import pandas as pd
from tqdm import tqdm

from transformers import get_cosine_schedule_with_warmup

from data_loader import *
from model import *
from helper_functions import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 32

trainset_subset = TweetDataset(train, tokenizer)
valset_subset = TweetDataset(test, tokenizer)

train_subset_sampler = BatchRandomSampler(np.arange(len(trainset_subset)), batch_size)
val_subset_sampler = BatchRandomSampler(np.arange(len(valset_subset)), batch_size)

trainloader = DataLoader(trainset_subset, batch_sampler=train_subset_sampler, collate_fn=custom_collate)
valloader = DataLoader(valset_subset, batch_sampler=val_subset_sampler, collate_fn=custom_collate)

model = AspectExtractor(distilbert).to(device)

lr=0.0001
EPOCHS = 15

optimizer = torch.optim.Adam(model.parameters(), lr=lr)

steps = EPOCHS * len(trainloader)
warmup_steps = len(trainloader)
scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=steps)


clipping_value = 1
train_index = pd.Index(data=range(len(trainloader)*EPOCHS), name='Batch')
val_index = pd.Index(data=range(len(trainloader)*EPOCHS), name='Batch')
columns = ['Epoch', 'Train Loss', 'Train Correct', 'Train Total']


for n,p in model.named_parameters():
    if(p.requires_grad) and ("bias" not in n):
        columns.append(n)
        

train_history = pd.DataFrame(columns=columns, index=train_index, data=0)
val_history = pd.DataFrame(columns=['Epoch', 'Val Loss', 'Val Correct', 'Val Total'], index=val_index, data=0)


train_batch = 0
val_batch = 0

for epoch in range(EPOCHS):
    
    trainloss = 0
    valloss = 0
    trainbar = tqdm(trainloader, unit=' Batch', desc=f'Train {epoch}', postfix='Loss=0.000', ncols='80%')
    
    model.train()

    
    for X, attn_mask, Y in trainbar:
        
        optimizer.zero_grad()
        
        X = X.to(device)
        attn_mask = attn_mask.to(device)
        Y = Y.to(device)
        
        loss, ncorrect = model.fit(X, attn_mask, Y)
        
        loss.backward()
        
        avg_grads = get_average_gradient(model.named_parameters())
        
        results = {'Epoch':epoch, 'Train Loss':loss.item(), 'Train Correct':ncorrect, 'Train Total':X.size(0)}
        
        results.update(avg_grads)
        
        results = pd.Series(name=train_batch, data=results)
    
        optimizer.step()
        scheduler.step()
     
        train_history.loc[train_batch, results.index] = results
        trainbar.set_postfix_str(f'Loss={train_history.loc[train_batch,"Train Loss"]:9.3f}')  
        
        train_batch += 1

    valbar = tqdm(valloader, unit=' Batch', desc=f'Val {epoch}', postfix='Loss=0.000', ncols='80%')

    model.eval()
    with torch.no_grad():
        for X, attn_mask, Y in valbar:
            
            X = X.to(device)
            attn_mask = attn_mask.to(device)
            Y = Y.to(device)
                      
            loss, ncorrect = model.fit(X, attn_mask, Y)

            results = {'Epoch':epoch, 'Val Loss':loss.item(), 'Val Correct':ncorrect, 'Val Total':X.size(0)}
            
            results = pd.Series(name=val_batch, data=results)
            
            val_history.loc[val_batch, results.index] = results
            
            valbar.set_postfix_str(f'Loss={val_history.loc[val_batch,"Val Loss"]:9.3f}')
        
            val_batch += 1