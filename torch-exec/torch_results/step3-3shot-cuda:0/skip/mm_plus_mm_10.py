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


input1 = torch.randn(3, 3)

input2 = torch.randn(3, 3)

input3 = torch.randn(3, 3)

input4 = torch.randn(3, 3)

test_inputs = [input1, input2, input3, input4]
