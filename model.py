import numpy as np
import torch
from torch import nn

class ClassificationHead(nn.Module):
    def __init__(self):
        super(ClassificationHead, self).__init__()
        
        self.dropout = nn.Dropout(0.2)
        self.ff = nn.Linear(in_features=768, out_features=256)
        self.activation = torch.nn.ReLU()
        self.output = nn.Linear(in_features=256, out_features=1)
        self.sigmoid = nn.Sigmoid()
    
    @staticmethod
    def init_weights(m):
        if type(m) == torch.nn.Linear:
            y = m.in_features
            m.weight.data.normal_(mean=0.0, std=1/np.sqrt(y))
            m.bias.data.fill_(0)
    
  
    def forward(self, X):
        X = self.dropout(X)
        X = self.ff(X)
        X = self.activation(X)
        X = self.output(X)
        X = self.sigmoid(X)
    
        return X
    
    def inference(self, X):
        predictions = self.forwad(X)
        pred_classes = self.get_class(predictions)
        return pred_classes
    
    def get_class(self, predictions):
        _, pred_classes = predictions.max(dim=1)
        return pred_classes
            
    
class AspectExtractor(nn.Module):
    def __init__(self, transformer):
        super(AspectExtractor, self).__init__()

        self.transformer = transformer
        self.head = ClassificationHead()
        self.criterion = nn.BCELoss(reduction = 'none')
        
        self.freeze()
 
    def forward (self, X, attn_mask):
        X = self.transformer(X, attn_mask)
        X = X[0]
        X = self.head(X)

        return X
       
    def fit(self, X, attn_mask, Y):
        
        logits = self.forward(X, attn_mask).squeeze()
        loss = self.criterion(logits, Y)
        
        weight = Y.sum(dim=1)/attn_mask.sum(dim=1)
        
        predictions = logits > 0.5
        ncorrect = (predictions == Y).all(dim=1).sum().item()
        
        return  loss, ncorrect
    
    def freeze(self):
        for param in self.transformer.parameters():
            param.requires_grad = False