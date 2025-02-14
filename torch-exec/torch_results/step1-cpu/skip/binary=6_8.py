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



func = ().to('cpu')


v1 = torch.rand(64, 64)

v2 = torch.rand(64, 64)

test_inputs = [v1, v2]
