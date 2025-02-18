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


x1 = torch.zeros(1, 1, 2)

test_inputs = [x1]
