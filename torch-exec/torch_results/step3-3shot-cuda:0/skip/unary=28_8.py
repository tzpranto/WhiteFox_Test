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

    def __init__(self, min_value=1.0, max_value=2.0):
        super().__init__()

    def forward(self, x1):
        v1 = x1
        v2 = torch.clamp_min(v1, min_value)
        v3 = torch.clamp_max(v2, max_value)
        return v3


func = Model(**parameters).to('cuda:0')

x1 = 1

test_inputs = [x1]
