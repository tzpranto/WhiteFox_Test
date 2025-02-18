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
        norm = torch.nn.BatchNorm2d(10).eval()
        norm.num_batches_tracked = 2
        norm.running_mean = torch.ones(10).to(torch.float16)
        norm.running_mean = norm.running_mean.to(torch.float32)
        norm.running_var = (torch.ones(10) * 2).to(torch.float16)
        norm.running_var = norm.running_var.to(torch.float32)
        c = torch.nn.Conv2d(10, 10, 3).eval()
        self.layer = torch.nn.Sequential(norm, c)

    def forward(self, x):
        a = self.layer(x)
        return a



func = Model().to('cpu')



x = torch.randn(5, 10, 100, 100, dtype=torch.float16)

test_inputs = [x]
