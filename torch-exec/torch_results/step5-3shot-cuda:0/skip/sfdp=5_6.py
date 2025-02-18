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
        self.__unfused_attn = torch.nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, x1, x2, __mask__):
        (v1, v2) = self.__unfused_attn(x1, x2, x2, key_padding_mask=__mask__)
        return v2


func = Model().to('cuda:0')


x1 = torch.randn(1, 5, 64, 64)

x2 = torch.randn(5, 8, 25)

mask = torch.zeros(5, 5)

test_inputs = [x1, x2, mask]
