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

    def __init__(self, batch_size=1, hidden=8, n_seq=10, n_head=2, hidden_per_head=4, dropout=0.5):
        super().__init__()
        self.multi_head_attntion = nn.MultiheadAttention(self.hparams.hidden, self.hparams.n_head, dropout=dropout)

    def forward(self, x, attn_mask=None):
        (attn_output, _) = self.multi_head_attntion(x, x, x, attn_mask)
        return attn_output


func = Model().to('cpu')


x = torch.randn(1, 10, 8)

attn_mask = torch.randn(1, 10, 10)

test_inputs = [x, attn_mask]
