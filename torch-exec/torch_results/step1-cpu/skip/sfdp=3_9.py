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

    def __init__(self, hparams: argparse.Namespace):
        super().__init__()
        self.d_q = hparams.d_q
        self.d_k = hparams.d_k
        self.softmax_temp = hparams.softmax_temp
        self.d_v = hparams.d_v
        self.d_model = hparams.d_model
        self.dropout = hparams.dropout
        self.scale_factor = math.sqrt(self.d_k)
        self.q_linear = torch.nn.Linear(hparams.d_input, self.d_q)
        self.k_linear = torch.nn.Linear(hparams.d_input, self.d_k)
        self.v_linear = torch.nn.Linear(hparams.d_input, self.d_v)
        self.projection = torch.nn.Linear(self.d_v, self.d_model)

    def forward(self, x):
        q = self.q_linear(x)
        k = self.k_linear(x)
        v = self.v_linear(x)
        scores = torch.matmul(q, k.transpose(-2, -1)).mul(self.scale_factor).softmax(dim=-1)
        p_attn = F.dropout(scores, p=self.dropout, training=self.training)
        context = torch.matmul(p_attn, v)
        output = self.projection(context)
        return (output, p_attn)


hparams = argparse.Namespace()
func = Model(hparams).to('cpu')


x = torch.randn(100, 40)

test_inputs = [x]
