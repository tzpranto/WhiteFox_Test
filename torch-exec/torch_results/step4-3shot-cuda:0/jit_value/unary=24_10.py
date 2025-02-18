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

    def __init__(self, negative_slope=0.01):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 1, stride=1, padding=1)
        self.negative_slope = negative_slope

    def forward(self, x1):
        v1 = self.conv(x1)
        v2 = v1 > 0
        v3 = v1 * self.negative_slope
        v4 = torch.where(v2, v2, v3)
        return v4



func = Model().to('cuda:0')


x1 = torch.randn(1, 3, 64, 64)

test_inputs = [x1]

# JIT_VALUE
'''
direct:
tensor([[[[ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          ...,
          [ 1.0000e+00, -8.6767e-03, -1.7859e-03,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00]],

         [[-2.4581e-03, -2.4581e-03, -2.4581e-03,  ..., -2.4581e-03,
           -2.4581e-03, -2.4581e-03],
          [-2.4581e-03, -1.2637e-02,  1.0000e+00,  ..., -4.1637e-03,
           -7.4738e-04, -2.4581e-03],
          [-2.4581e-03, -6.7119e-03, -7.6041e-03,  ..., -1.4575e-03,
           -8.9656e-03, -2.4581e-03],
          ...,
          [-2.4581e-03, -1.1798e-03,  1.0000e+00,  ..., -5.3827e-03,
           -3.9094e-03, -2.4581e-03],
          [-2.4581e-03, -2.6901e-03,  1.0000e+00,  ..., -7.6380e-03,
           -1.4767e-02, -2.4581e-03],
          [-2.4581e-03, -2.4581e-03, -2.4581e-03,  ..., -2.4581e-03,
           -2.4581e-03, -2.4581e-03]],

         [[-2.7327e-03, -2.7327e-03, -2.7327e-03,  ..., -2.7327e-03,
           -2.7327e-03, -2.7327e-03],
          [-2.7327e-03,  1.0000e+00, -6.1332e-03,  ..., -5.1805e-04,
           -2.0326e-02, -2.7327e-03],
          [-2.7327e-03, -1.5880e-03, -1.8807e-03,  ..., -9.0438e-03,
           -5.7294e-03, -2.7327e-03],
          ...,
          [-2.7327e-03,  1.0000e+00, -3.9241e-03,  ...,  1.0000e+00,
            1.0000e+00, -2.7327e-03],
          [-2.7327e-03, -3.9669e-03, -7.5431e-03,  ..., -2.3543e-03,
            1.0000e+00, -2.7327e-03],
          [-2.7327e-03, -2.7327e-03, -2.7327e-03,  ..., -2.7327e-03,
           -2.7327e-03, -2.7327e-03]],

         ...,

         [[ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00, -9.7301e-03,  1.0000e+00,  ..., -9.3073e-04,
           -3.2987e-03,  1.0000e+00],
          [ 1.0000e+00, -4.3151e-03, -5.4791e-03,  ..., -4.1932e-05,
           -7.6905e-03,  1.0000e+00],
          ...,
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00, -1.6364e-04,  1.0000e+00,  ..., -5.5348e-03,
           -1.2567e-02,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00]],

         [[ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          ...,
          [ 1.0000e+00, -2.4971e-03,  1.0000e+00,  ..., -2.7897e-03,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
           -2.8089e-03,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00]],

         [[ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00, -1.3268e-03,  ...,  1.0000e+00,
           -1.7725e-02,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ..., -4.8596e-03,
           -1.8085e-03,  1.0000e+00],
          ...,
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00, -9.1507e-04, -2.0759e-03,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00]]]], device='cuda:0')

jit:
tensor([[[[ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          ...,
          [ 1.0000e+00, -8.6758e-03, -1.7846e-03,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00]],

         [[-2.4581e-03, -2.4581e-03, -2.4581e-03,  ..., -2.4581e-03,
           -2.4581e-03, -2.4581e-03],
          [-2.4581e-03, -1.2638e-02,  1.0000e+00,  ..., -4.1637e-03,
           -7.4307e-04, -2.4581e-03],
          [-2.4581e-03, -6.7123e-03, -7.6056e-03,  ..., -1.4577e-03,
           -8.9679e-03, -2.4581e-03],
          ...,
          [-2.4581e-03, -1.1782e-03,  1.0000e+00,  ..., -5.3816e-03,
           -3.9096e-03, -2.4581e-03],
          [-2.4581e-03, -2.6903e-03,  1.0000e+00,  ..., -7.6388e-03,
           -1.4769e-02, -2.4581e-03],
          [-2.4581e-03, -2.4581e-03, -2.4581e-03,  ..., -2.4581e-03,
           -2.4581e-03, -2.4581e-03]],

         [[-2.7327e-03, -2.7327e-03, -2.7327e-03,  ..., -2.7327e-03,
           -2.7327e-03, -2.7327e-03],
          [-2.7327e-03,  1.0000e+00, -6.1338e-03,  ..., -5.1843e-04,
           -2.0327e-02, -2.7327e-03],
          [-2.7327e-03, -1.5880e-03, -1.8807e-03,  ..., -9.0440e-03,
           -5.7279e-03, -2.7327e-03],
          ...,
          [-2.7327e-03,  1.0000e+00, -3.9249e-03,  ...,  1.0000e+00,
            1.0000e+00, -2.7327e-03],
          [-2.7327e-03, -3.9667e-03, -7.5454e-03,  ..., -2.3530e-03,
            1.0000e+00, -2.7327e-03],
          [-2.7327e-03, -2.7327e-03, -2.7327e-03,  ..., -2.7327e-03,
           -2.7327e-03, -2.7327e-03]],

         ...,

         [[ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00, -9.7292e-03,  1.0000e+00,  ..., -9.3038e-04,
           -3.2996e-03,  1.0000e+00],
          [ 1.0000e+00, -4.3154e-03, -5.4810e-03,  ..., -4.4137e-05,
           -7.6937e-03,  1.0000e+00],
          ...,
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00, -1.6433e-04,  1.0000e+00,  ..., -5.5354e-03,
           -1.2567e-02,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00]],

         [[ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          ...,
          [ 1.0000e+00, -2.4974e-03,  1.0000e+00,  ..., -2.7867e-03,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
           -2.8083e-03,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00]],

         [[ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00, -1.3272e-03,  ...,  1.0000e+00,
           -1.7725e-02,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ..., -4.8603e-03,
           -1.8090e-03,  1.0000e+00],
          ...,
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00, -9.1472e-04, -2.0765e-03,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00],
          [ 1.0000e+00,  1.0000e+00,  1.0000e+00,  ...,  1.0000e+00,
            1.0000e+00,  1.0000e+00]]]], device='cuda:0')
'''