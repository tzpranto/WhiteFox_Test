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


query = torch.randn(1, 1, 1, 1)

key = torch.randn(1, 1, 1, 1)

test_inputs = [query, key]
