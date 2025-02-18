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
        self.head = torch.nn.TransformerEncoderLayer(4, 4, 4, 4, 2)

    def forward(self, x1):
        v1 = self.head(x1)
        return v1


func = Model().to('cpu')


x1 = torch.randn(3, 5, 4)

test_inputs = [x1]
