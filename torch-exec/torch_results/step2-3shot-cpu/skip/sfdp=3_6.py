import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import math
import torch as th
import torch.linalg as la
from torch.nn import Parameter
import torch.linalg as linalg

class Model(torch.nn.Module):

    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads):
        super().__init__()
        self.encoder = torch.nn.TransformerEncoder(torch.nn.TransformerEncoderLayer(hidden_size, num_attention_heads, hidden_dropout=0.0, batch_first=True), num_hidden_layers)

    def forward(self, x):
        v = self.encoder(x)
        return v


hidden_size = 1
num_hidden_layers = 1
num_attention_heads = 1

func = Model(hidden_size, num_hidden_layers, num_attention_heads).to('cpu')


x = torch.randn(3, 3, 64)

test_inputs = [x]
