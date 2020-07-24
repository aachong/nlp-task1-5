import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import cnews_dataset

class avgModel(nn.Module):
    def __init__(self,max_vocab_size,embedding_size,dropout,label_size):
        super(avgModel,self).__init__()
        self.embedding = nn.Embedding(max_vocab_size,embedding_size)
        self.l1 = nn.Linear(embedding_size,label_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self,input:torch.tensor):
        #input :n*l
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        #embeded:n*l*e
        c1 = F.avg_pool2d(embedded,(embedded.shape[1],1)).squeeze()
        c2 = F.max_pool2d(embedded,(embedded.shape[1],1)).squeeze()
        #n*e
        pooled = c1+c2
        return self.l1(pooled)#n*10

if __name__ == '__main__':
    pass
    

