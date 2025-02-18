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

    def __init__(self, args):
        super().__init__()
        self.w_q = torch.nn.Linear(QUERY_LEN, KEY_LEN)

    def forward(self):
        qk = self.w_q(query_input)
        scaled_qk = qk.mul(scale_factor)
        softmax_qk = scaled_qk.softmax(dim=-1)
        dropout_qk = torch.nn.functional.dropout(softmax_qk, p=dropout_p)
        output = dropout_qk.matmul(value)
        return output


args = 1

func = Model(args).to('cpu')

input_tensor = torch.randn(1, 1, 1)

test_inputs = [input_tensor]
