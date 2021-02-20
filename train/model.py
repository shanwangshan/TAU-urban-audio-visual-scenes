import torch.nn as nn
import torch
'''
This script is part of train.py. We define our model as a class
'''


class l3_dense(nn.Module):
    def __init__(self,emb_dim,num_classes):
        super(l3_dense, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim        
        self.model = nn.Sequential(
          nn.Linear(self.emb_dim,512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(512,128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(128,64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(64,self.num_classes)
        )
    def forward(self, x):
        y = self.model(x) # shape is [batch_size , 512]
        return y

'''
class l3_audio_dense(nn.Module): #########model is saved model_1.pt
    def __init__(self,emb_dim,num_classes):
        super(l3_audio_dense, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim        
        self.model = nn.Sequential(
          nn.Linear(self.emb_dim,512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(512,128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(128,self.num_classes)
        )
    def forward(self, x):
        y = self.model(x) # shape is [batch_size , 512]
        return y
'''
'''
class l3_audio_dense(nn.Module):
    def __init__(self,emb_dim,num_classes):
        super(l3_audio_dense, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim        
        self.model = nn.Sequential(
          nn.Linear(self.emb_dim,64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Dropout(p=0.2),
          nn.Linear(64,self.num_classes)
        )
    def forward(self, x):
        y = self.model(x) # shape is [batch_size , 512]
        return y
'''