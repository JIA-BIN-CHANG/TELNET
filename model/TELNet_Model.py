# -*- coding: utf-8 -*-
"""
Created on Fri May 20 10:41:23 2022

@author: USER
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import layer_norm as ln

class SelfAttention(nn.Module):

    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.5)

    def forward(self, x):
        n = x.shape[0]  # sequence length

        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        V = self.V(x)

        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1,0))
        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(weights,V)
        y = self.output_linear(y)

        return y

class TELNet_model(nn.Module):
    
    def __init__(self,input_size=4096,windowSize=15):
        super(TELNet_model, self).__init__()
        
        self.att = SelfAttention(input_size=4096)
        self.linear1 = nn.Linear(in_features=input_size, out_features=1024)
        self.linear2 = nn.Linear(in_features=1024, out_features=windowSize)
        self.linear3 = nn.Linear(in_features=1024, out_features=1024)
        self.linear4 = nn.Linear(in_features=1024, out_features=windowSize)
        
        self.sig = nn.Sigmoid()
        self.drop50 = nn.Dropout(0.5)
        self.layer_norm1 = ln.LayerNorm(input_size)
        self.layer_norm2 = ln.LayerNorm(1024)
        self.layer_norm3 = ln.LayerNorm(1024)
    
    def forward(self, feature):
        # Self Attention layer
        att_out = self.att(feature)
        y = att_out + feature # Add Residual 
        y = self.drop50(y)
        y = self.layer_norm1(y)
        
        # Two linear layer classifier (Linker)
        y = self.linear1(y)
        y = F.relu(y)
        y = self.drop50(y)
        y = self.layer_norm2(y)
        
        y = self.linear2(y)
        y = self.sig(y)
        
        y = y.view(1,-1)
        
        return y
    
if __name__ == '__main__':
    pass
        
        
        