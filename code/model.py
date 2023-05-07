from typing import DefaultDict
import numpy as np
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import roc_auc_score,log_loss


import os
import random

from utils import Config

class BaseModel(nn.Module):
    def __init__(self, sizes, hidden_factors=EMD_DIM):
        super(BaseModel, self).__init__()
        self.num_feat = len(sizes)
        self.hidden_factors = hidden_factors
        self.embeddings = torch.nn.ModuleList([
            nn.Embedding(sizes[i]+1, hidden_factors) for i in range(len(sizes))
        ])
        hid_dim=HID_DIM
        
        self.deep_layers = nn.Sequential(
            nn.Linear(hidden_factors*len(sizes), hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            # nn.Linear(hid_dim, hid_dim, bias=False),
            # nn.BatchNorm1d(hid_dim),
            # nn.ReLU(),
            # nn.Dropout(0.5),
            nn.Linear(hid_dim, 1, bias=False),
            nn.Sigmoid()
        )
        
        for layer in self.deep_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
        
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, std=0.01)
        
    def forward(self, features):
        embeds = [self.embeddings[i](features[:,i]) for i in range(self.num_feat)]
        output = self.deep_layers(torch.cat(embeds,dim=1))
        return output

    def reg(self,guards,embeds):
        l = 0
        # print(torch.norm(self.embeddings[1](3))
        for feat in range(num_categorial_feat):
            g = guards[feat].to(device)
            new_embed = self.embeddings[feat](g)
            old_embed = torch.index_select(embeds[feat],0,g)
            l_temp=g*torch.norm(new_embed-old_embed,dim=1)
            l+=torch.sum(l_temp,dim=0)
        return l
        
        