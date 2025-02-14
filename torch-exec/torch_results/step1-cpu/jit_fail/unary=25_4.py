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
        self.linear = torch.nn.Linear(3, 8)

    def forward(self, x):
        v1 = self.linear(x)
        v2 = v1.gt(0)
        v3 = v2.int()
        v4 = v3.add(2)
        v5 = v4.long()
        v6 = v5.sub(1)
        v7 = v6 != 2
        v8 = v7 | v2
        v9 = v8.float()
        v10 = v8.lt(0)
        v11 = v10 - 0.5
        v12 = v9 & v11
        v13 = v12.sum(0)
        v14 = v13.exp()
        output = v14.log()
        return output


func = Model().to('cpu')


x = torch.randn(1, 3)

test_inputs = [x]

# JIT_FAIL
'''
direct:
Subtraction, the `-` operator, with a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.

jit:
backend='inductor' raised:
CppCompileError: C++ compile error

Command:
/scratch/mpt5763/miniconda3/envs/wf/bin/x86_64-conda-linux-gnu-c++ /tmp/torchinductor_mpt5763/c6/cc6e27e5infc5tedhejhh5ketmmvlvypzpmwvg3sjdqlro5lr5u2.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX512 -shared -fPIC -O3 -DNDEBUG -ffast-math -fno-finite-math-only -fno-unsafe-math-optimizations -ffp-contract=off -march=native -Wall -std=c++17 -Wno-unused-variable -Wno-unknown-pragmas -fopenmp -I/scratch/mpt5763/miniconda3/envs/wf/include/python3.9 -I/scratch/mpt5763/miniconda3/envs/wf/include/python3.9 -I/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/include -I/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/include/TH -I/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/include/THC -I/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/include -I/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/include/torch/csrc/api/include -I/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/include/TH -I/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/include/THC -mavx512f -mavx512dq -mavx512vl -mavx512bw -mfma -mamx-tile -mamx-bf16 -mamx-int8 -D_GLIBCXX_USE_CXX11_ABI=0 -ltorch -ltorch_cpu -ltorch_python -lc10 -lgomp -L/scratch/mpt5763/miniconda3/envs/wf/lib -L/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/lib -L/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/lib -L/scratch/mpt5763/miniconda3/envs/wf/lib/python3.9/site-packages/torch/lib -o /tmp/torchinductor_mpt5763/c6/cc6e27e5infc5tedhejhh5ketmmvlvypzpmwvg3sjdqlro5lr5u2.so

Output:
/tmp/torchinductor_mpt5763/c6/cc6e27e5infc5tedhejhh5ketmmvlvypzpmwvg3sjdqlro5lr5u2.cpp: In function 'void kernel(const float*, const float*, const float*, float*)':
/tmp/torchinductor_mpt5763/c6/cc6e27e5infc5tedhejhh5ketmmvlvypzpmwvg3sjdqlro5lr5u2.cpp:46:48: error: invalid operands of types 'float' and 'float' to binary 'operator&'
   46 |             auto tmp34 = decltype(tmp27)(tmp27 & tmp33);
      |                                          ~~~~~ ^ ~~~~~
      |                                          |       |
      |                                          float   float



You can suppress this exception and fall back to eager by setting:
    import torch._dynamo
    torch._dynamo.config.suppress_errors = True

'''