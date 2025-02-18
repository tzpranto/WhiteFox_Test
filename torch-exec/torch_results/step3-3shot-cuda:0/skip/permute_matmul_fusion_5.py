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



func = ().to('cuda:0')


x = torch.randn(1, 2, 2)
x = torch.randn(1, 2, 2)
a = x[0:1].permute(0, 2, 1)

a = x[0:1].permute(0, 2, 1)
b = x[0:1]
y = torch.matmul(b, a)

test_inputs = [x, a, y]
