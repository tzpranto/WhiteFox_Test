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


query = torch.randn(1, 5, 128)

key = torch.randn(1, 5, 128)

attn_mask = torch.randn(1, 5)

test_inputs = [query, key, attn_mask]
