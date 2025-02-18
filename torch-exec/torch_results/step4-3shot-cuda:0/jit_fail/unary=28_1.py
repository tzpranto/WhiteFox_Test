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

    def forward(self, input_tensor):
        l = torch.nn.Linear(5, 5)
        output = input_tensor * torch.clamp_min(torch.nn.functional.linear(input_tensor, l.weight, l.bias), min_value=-0.5)
        return torch.clamp(torch.tanh(output), max_value=0.05)



func = Model().to('cuda:0')


input0 = torch.randn(3, 5)

test_inputs = [input0]

# JIT_FAIL
'''
direct:
Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0! (when checking argument for argument mat1 in method wrapper_CUDA_addmm)

jit:
Failed running call_function <built-in function linear>(*(FakeTensor(..., device='cuda:0', size=(3, 5)), Parameter(FakeTensor(..., size=(5, 5), requires_grad=True)), Parameter(FakeTensor(..., size=(5,), requires_grad=True))), **{}):
Unhandled FakeTensor Device Propagation for aten.mm.default, found two different devices cuda:0, cpu

from user code:
   File "<string>", line 20, in torch_dynamo_resume_in_forward_at_19


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''