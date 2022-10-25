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


class PositionalEncoding(nn.Module):
    """
    Positional Encodeing, 這部分聽說很重要但我自己也沒搞懂
    有沒有影響我真的不知道，可省略
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
    
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


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

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)
            logits[torch.eye(n).byte()] = -float("Inf")

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")

        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
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
        self.drop25 = nn.Dropout(0.25)
        self.layer_norm1 = ln.LayerNorm(input_size)
        self.layer_norm2 = ln.LayerNorm(1024)
        self.layer_norm3 = ln.LayerNorm(1024)
    def forward(self, feature):
        # Self Attention layer    org
        att_out = self.att(feature)
        y = att_out + feature # Add Residual 
        y = self.drop50(y)
        y = self.layer_norm1(y)
        
        # Two linear layer classifier
        y = self.linear1(y)
        y = F.relu(y)
        y = self.drop50(y)
        y = self.layer_norm2(y)
        
        y = self.linear2(y)
        y = self.sig(y)
        
        

      
        
        y = y.view(1,-1)
        
        return y

class MyTransformer(nn.Module):
    """
    TELNet 的架構
    說明:
        __init__()中的參數為建構時所需要給的參數
        d_model: 輸入feature的維度，以之前的例子來說是4096
        nhead: 這邊可參考Attention is all you need中的nhead說明(就只是一個參數啦，可以改改看有沒有影響)
        num_layer: 依樣可參考Attention is all you need(也只是一個參數)
        window_size: 如同我的paper中所提到，一次看幾個shot，之前是15
        
        forward()為訓練時要給的資料
        shot_feature: shot feature 為一個15x4096(window_size x d_model)的torch.tensor
        x: 15 x 15torch.tensor 沒意外的話是任意shot對每一個shot的分數
    """
    def __init__(self,d_model,nhead,num_layer,windowSize):
        super(MyTransformer,self).__init__()
        ##Model Parameter
        self.d_model = d_model
        self.nhead = nhead
        self.num_lay = num_layer
        #Model Architecture
        self.pe = PositionalEncoding(d_model)
        encoderLayer = nn.TransformerEncoderLayer(d_model, nhead)
        self.encoder = nn.TransformerEncoder(encoderLayer, num_layer)
        self.layer1 = nn.Linear(d_model,2048)
        self.layer2 = nn.Linear(2048,1024)
        self.layer3 = nn.Linear(1024,windowSize)
        
    def forward(self,shot_feature):
        num_shot = shot_feature.shape[0]
        src = shot_feature.view(num_shot,1,self.d_model)
        src = self.pe(src)
        attention_out = self.encoder(src)
        x = F.relu(self.layer1(attention_out))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        
        return x
    
if __name__ == '__main__':
    pass
        
        
        