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

    def __init__(self, config, input_shape):
        super().__init__()
        self.proj = torch.nn.Linear(input_shape, config.hidden_size)

    def forward(self, x):
        x = self.proj(x)
        return x


config = CONFIG['attention']()
input_shape = (1, 4, 20)
func = Model(config, input_shape).to('cpu')


input_shape = (1, 4, 20)
x = torch.randn(input_shape)

test_inputs = [x]
