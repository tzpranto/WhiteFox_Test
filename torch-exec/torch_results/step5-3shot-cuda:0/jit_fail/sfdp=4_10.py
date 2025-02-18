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

    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, dropout=0.2)

    def forward(self, x1):
        (output, attention) = self.attn(x1, x1, x1, attn_mask=None, key_padding_mask=None)[0]
        return output


func = Model().to('cuda:0')


x1 = torch.randn(27, 8, 64)

test_inputs = [x1]

# JIT_FAIL
'''
direct:
was expecting embedding dimension of 8, but got 64

jit:
Failed running call_function <function multi_head_attention_forward at 0x7fab6d6855e0>(*(FakeTensor(..., device='cuda:0', size=(27, 8, 64)), FakeTensor(..., device='cuda:0', size=(27, 8, 64)), FakeTensor(..., device='cuda:0', size=(27, 8, 64)), 8, 2, Parameter(FakeTensor(..., device='cuda:0', size=(24, 8), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(24,), requires_grad=True)), None, None, False, 0.2, Parameter(FakeTensor(..., device='cuda:0', size=(8, 8), requires_grad=True)), Parameter(FakeTensor(..., device='cuda:0', size=(8,), requires_grad=True))), **{'training': False, 'key_padding_mask': None, 'need_weights': True, 'attn_mask': None, 'average_attn_weights': True, 'is_causal': False}):
was expecting embedding dimension of 8, but got 64

from user code:
   File "<string>", line 20, in forward
  File "/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/nn/modules/activation.py", line 1368, in forward
    attn_output, attn_output_weights = F.multi_head_attention_forward(


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''