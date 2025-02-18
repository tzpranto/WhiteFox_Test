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

    def __init__(self, num_layers):
        super().__init__()
        self.pos_emb = torch.nn.Parameter(torch.empty([1, 2304, 14, 14]))
        self.transformer_layers = torch.nn.ModuleList([transformer.TransformerBlock(2304, 1024, 512) for _ in range(num_layers)])

    def forward(self, input_tensor):
        x = input_tensor + self.pos_emb
        for transformer in self.transformer_layers:
            x = transformer(x)
        return x


num_layers = 1

func = Model(num_layers).to('cuda:0')


x = torch.randn(3, 3, 224, 224)

test_inputs = [x]
