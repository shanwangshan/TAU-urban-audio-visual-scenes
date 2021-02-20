import torch.nn as nn
import torch

#from IPython import embed
# Here we define our model as a class
class l3_dense(nn.Module):

    def __init__(self,emb_dim,num_classes):

        super(l3_dense, self).__init__()
        #self.flag = flag
    
        self.num_classes = num_classes
        self.emb_dim = emb_dim        

        #self.layer_1 = nn.Linear(self.emb_dim, self.num_classes)   
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
                            # x shape os [batch_size, emb_dim]
        y = self.model(x) # shape is [batch_size , 512]
      
    
        return y
