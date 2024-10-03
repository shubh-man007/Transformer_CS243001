import torch
import torch.nn as nn
import numpy as np
import math

g = torch.Generator().manual_seed(2147483647)

class InputEmbeddings(nn.Module):
    def __init__(self,d_model : int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embeds = nn.Embedding(vocab_size,d_model)

    def forward(self,x:torch.tensor):  # x is the lookup table, denotes vocab position
        return self.embeds(x) * (self.d_model)**(0.5)
    

class PositionalEmbedding(nn.Module):
    def __init__(self,d_model : int, seq_len : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        #Dropout avoids overfitting by dropping random weights which helps in generalisation
        self.dropout = nn.Dropout(dropout)

        #Create a matrix of shape (seq_len,d_model)
        pos = torch.zeros(seq_len,d_model)
        #Create a vector of shape (seq_len,1)
        position = torch.arange(0,seq_len,dtype = torch.float).unsqueeze(1)
        denominator = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model))
        #Apply sin to even positions:
        pos[:,0::2] = torch.sin(position * denominator)
        #Apply cos to odd positions:
        pos[:,1:,2] = torch.cos(position * denominator)

        #unsqueeze : Returns a new tensor with a dimension of size one inserted at the specified position.
        pos = pos.unsqueeze(0)

        #Because this is a Non Trainable class, we save its values in the buffer to avoid repetitive calculations:
        self.register_buffer('pos',pos)

    def forward(self,x):
        #requires_grad(False) does not calculate the gradient during backpropagation because PE is Non Trainable
        x = x + (self.pos[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)
    

class LayerNorm(nn.Module):
    def __init__(self,epsilon : float = 10**-6):
        super().__init__()
         #Epsilon is used for numerical stability
        self.epsilon = epsilon
        #nn.Parameter turns the variable to be learnable
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self,x : torch.tensor):
        mean = torch.mean(x, dim = -1, keepdim = True)
        std = torch.std(x, dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (self.epsilon + std) + self.bias
    

class FeedForward(nn.Module):
     #d_ff has been specified in the paper, it specifies number if neurons in hidden layer of feed forward nn
    def __init__(self,d_model : int, dropout : float, d_ff : int = 2048):
        super().__init__()
        self.Layer1 = nn.Linear(d_model,d_ff)
        self.Layer2 = nn.Linear(d_ff,d_model)
        self.dropout = nn.Dropout(dropout)
        #ReLU is used to introduce non linearity in our model
        self.relu = nn.ReLU

    def forward(self,x : torch.tensor):
        return self.Layer2(self.dropout(self.relu(self.Layer1(x))))


class SelfAttention(nn.Module):
    def __init__(self,embed_stack : torch.tensor, d_model : int, n_head : int = 1):
        super().__init__()
        self.embed_stack = embed_stack
        self.d_model = d_model
        self.n_heads = n_head
        assert d_model % n_head == 0, "d_model is not divisible by h"
        self.w_q = torch.randn((d_model,d_model/n_head))      
        self.w_k = torch.randn((d_model,d_model/n_head))
        self.w_v = torch.randn((d_model,d_model/n_head))

    def __call__(self,mask):
        Q = self.embed_stack @ self.w_q
        K = self.embed_stack @ self.w_k
        V = self.embed_stack @ self.w_v
        K_T = torch.transpose(K,0,1)
        arg = (Q @ K_T)/math.sqrt(self.d_model/self.n_head)
        if mask is not None:
            arg.masked_fill_(mask == 0, -1e9)
        out_soft = torch.softmax(arg,dim = 1)
        new_embed = out_soft @ V
        return new_embed, arg


class MultiHeadAttention(nn.Module,SelfAttention):
    def __init__(self, embed_stack : torch.tensor, d_model : int, n_head : int, ):
        super().__init__()
        self.embed_stack = embed_stack
        self.d_model = d_model
        self.n_heads = n_head
        self.heads = [SelfAttention(embed_stack,d_model,n_head) for _ in n_head]

    def forward(self):
        outs = [attention() for attention in self.heads]
        outs = torch.tensor(outs)
        return outs


class ResidualConnection(nn.Module): #skip connections
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNorm(features)
    
        def forward(self, x, sublayer):
            return x + self.dropout(sublayer(self.norm(x)))