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

    def __init__(self, other):
        super().__init__()
        self.conv = torch.nn.Conv2d(3, 8, 3, stride=1, padding=1)
        self.other = torch.nn.Parameter(other.reshape(1, 1, 1, 1))

    def forward(self, x):
        v1 = self.conv(x)
        v2 = self.other
        v3 = v2 - v1
        v4 = np.array([[-1.027756, -0.002286639, -0.001510427, -0.0009867046][-0.004274235, -0.01359701, -0.01104885, -0.00629437][-9.07233e-08, 7.456594e-08, 0.01389359, 0.002382906][3.021077e-07, -2.305938e-07, 0.006808552, 0.0007897332][0.00581536, 0.01292148, 0.233848, 0.8578209][0.01520595, 0.003752044, 0.0007897332, 0.1060929][-0.01104885, -0.006238518, -0.01636302, -0.005617751][-0.004738528, -0.009018803, -0.2910816, -0.6505371]]).reshape(1, 1, 8, 8)
        v4 = torch.from_numpy(v4)
        v5 = v3 + v4
        v6 = torch.nn.ReLU()(v5)
        return v6


other = np.array([-0.12918424, -0.21271498, -0.06773838, 0.04655406]).reshape(1, 4, 1, 1)
func = Model(other).to('cpu')

x = 1

test_inputs = [x]
