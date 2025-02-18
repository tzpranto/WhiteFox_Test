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

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.lin1 = nn.Linear(4, 4)
        self.lin2 = nn.Linear(4, 4)
        self.lin3 = nn.Linear(4, 4)

    def forward(self, input1, input2, input3, input4):
        input1 = torch.matmul(self.lin1(input1), torch.tensor([[[1, 2, 3, 4]]]).float())
        input2 = torch.matmul(input2, torch.tensor([[[2, 3, 1, 4]]]).float())
        input3 = torch.matmul(self.lin2(input3), torch.tensor([[[2, 3, 4, 1]]]).float())
        input4 = torch.matmul(input4, torch.tensor([[[2, 3, 4, 1]]]).float())
        output = torch.add(input1, input2)
        output = torch.add(output, input3)
        output = torch.add(output, input4)
        return output



func = Model().to('cuda:0')


input1 = torch.rand(3, 4)

input2 = torch.rand(3, 4)

input3 = torch.rand(3, 4)

input4 = torch.rand(3, 4)

test_inputs = [input1, input2, input3, input4]

# JIT_FAIL
'''
direct:
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument mat2 in method wrapper_CUDA_bmm)

jit:
Failed running call_function <built-in method matmul of type object at 0x7efd99a5f1c0>(*(FakeTensor(..., device='cuda:0', size=(3, 4)), FakeTensor(..., size=(1, 1, 4))), **{}):
Expected size for first two dimensions of batch2 tensor to be: [1, 4] but got: [1, 1].

from user code:
   File "<string>", line 22, in forward


You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''