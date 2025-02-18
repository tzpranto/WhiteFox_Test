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

class model(torch.nn.module):

    def __init__(self):
        super().__init__()
        self.x = torch.rand(8, requires_grad=True)

    def forward(self, x):
        v1 = torch.nn.functional.dropout(self.x, p=0.2)
        v2 = torch.rand_like(x)
        return v1 + v2



func = model().to('cuda:0')


x1 = torch.randn(1, 2)

test_inputs = [x1]
