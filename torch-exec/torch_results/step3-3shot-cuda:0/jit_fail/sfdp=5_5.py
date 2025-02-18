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

    def __init__(self):
        super().__init__()
        self.trans_layer_norm = torch.nn.LayerNorm(768, eps=1e-06)
        self.self_attention = torch.nn.MultiheadAttention(768, 4, dropout=0.2, bias=True)
        self.trans_fc = torch.nn.Linear(768, 3072)

    def forward(self, x2):
        v2 = self.query



func = Model().to('cuda:0')

x2 = 1

test_inputs = [x2]

# JIT_FAIL
'''
direct:
'Model' object has no attribute 'query'

jit:
'Model' object has no attribute 'query'
'''