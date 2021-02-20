import torch.nn as nn
import torch


class l3_combine(nn.Module): #########model_2.pt#####

    def __init__(self,emb_dim,num_classes):
        super(l3_combine, self).__init__()
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.model = nn.Sequential(
          nn.Linear(self.emb_dim,128),

            nn.Linear(128,self.num_classes),
        )
    def forward(self, x):
        y = self.model(x)
        return y
