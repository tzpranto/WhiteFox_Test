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

    def __init__(self, emb, heads, dropout_p):
        super().__init__()
        self.emb = emb
        self.heads = heads
        self.scale_factor = emb ** 0.5
        self.dropout_p = dropout_p
        self.k_conv = torch.nn.Conv2d(3, emb, 1, stride=1, padding=1)
        self.q_conv = torch.nn.Conv2d(3, emb, 1, stride=1, padding=1)
        self.v_conv = torch.nn.Conv2d(3, emb, 1, stride=1, padding=1)
        self.o_conv = torch.nn.Conv2d(emb, emb, 1, stride=1, padding=1)

    def forward(self, x1, x2):
        k = self.k_conv(x1)
        q = self.q_conv(x2)
        v = self.v_conv(x1)
        scale_factor = self.scale_factor
        dropout_p = self.dropout_p
        qk = torch.matmul(q, k.transpose(-2, -1))
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(v)


emb = 1
heads = 1
dropout_p = 1

func = Model(emb, heads, dropout_p).to('cuda:0')

x1 = 1
x2 = 1

test_inputs = [x1, x2]

# JIT_FAIL
'''
direct:
conv2d() received an invalid combination of arguments - got (int, Parameter, Parameter, tuple, tuple, tuple, int), but expected one of:
 * (Tensor input, Tensor weight, Tensor bias = None, tuple of ints stride = 1, tuple of ints padding = 0, tuple of ints dilation = 1, int groups = 1)
      didn't match because some of the arguments have invalid types: (!int!, !Parameter!, !Parameter!, !tuple of (int, int)!, !tuple of (int, int)!, !tuple of (int, int)!, !int!)
 * (Tensor input, Tensor weight, Tensor bias = None, tuple of ints stride = 1, str padding = "valid", tuple of ints dilation = 1, int groups = 1)
      didn't match because some of the arguments have invalid types: (!int!, !Parameter!, !Parameter!, !tuple of (int, int)!, !tuple of (int, int)!, !tuple of (int, int)!, !int!)


jit:
conv2d() received an invalid combination of arguments - got (int, Parameter, Parameter, tuple, tuple, tuple, int), but expected one of:
 * (Tensor input, Tensor weight, Tensor bias = None, tuple of ints stride = 1, tuple of ints padding = 0, tuple of ints dilation = 1, int groups = 1)
      didn't match because some of the arguments have invalid types: (!int!, !Parameter!, !Parameter!, !tuple of (int, int)!, !tuple of (int, int)!, !tuple of (int, int)!, !int!)
 * (Tensor input, Tensor weight, Tensor bias = None, tuple of ints stride = 1, str padding = "valid", tuple of ints dilation = 1, int groups = 1)
      didn't match because some of the arguments have invalid types: (!int!, !Parameter!, !Parameter!, !tuple of (int, int)!, !tuple of (int, int)!, !tuple of (int, int)!, !int!)

'''