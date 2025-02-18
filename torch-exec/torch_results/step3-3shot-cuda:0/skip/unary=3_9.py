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
        self.t = torch.Tensor([97, 7, 4, 129, 3, 17, 85, 86, 95, 123, 57, 113, 48, 36, 22, 147, 54, 118, 67, 142, 127, 4, 10, 41, 95, 1, 8, 17, 22, 11, 6, 70, 50, 104, 149, 76, 134, 84, 36, 8, 102, 70, 39, 83, 48, 144, 144, 35, 20, 141, 144, 15, 48, 81, 15, 128, 48, 54, 109, 105, 63, 67, 72, 18, 48, 103, 133, 88, 60, 29, 73, 3, 112, 27, 65, 134, 70, 44, 72, 85, 116, 71, 15, 60, 127, 30, 17, 76, 115, 134, 81, 140, 141, 92, 41, 101, 45, 131, 145, 72, 64, 28, 103, 55, 6, 79, 27, 38, 15, 77, 83, 40, 23, 116, 17, 55, 46, 123, 144, 106, 124, 87, 57, 41, 21, 139, 150, 72, 131, 21, 148, 32, 43, 42, 94, 67, 114, 3, 35, 72, 23, 69, 49, 96, 147, 71, 68, 56, 111, 34, 17, 28, 146, 68, 131, 49, 44, 99, 149, 131, 97, 109, 130, 8, 128, 132, 35, 85, 16, 120, 77, 6, 46, 103, 14, 1, 134, 149, 115, 20, 80, 143, 46, 118, 130, 55, 111, 121, 26, 133, 134, 87, 35, 124, 79, 111, 2, 112, 35, 107, 11, 4, 40, 87, 119, 0, 111, 116, 149, 145, 144, 45, 115, 13, 137, 138, 78, 46, 24, 57, 133, 40, 21, 112, 138, 34, 147, 125])
        self.t = self.t.reshape([81, 1, 7, 7])
        self.conv1 = torch.nn.Conv2d(113, 2, 7, stride=7, padding=0)
        self.conv1.weight = torch.nn.Parameter(self.t)

    def forward(self, x1):
        v1 = self.conv1(x1)
        return v1



func = Model().to('cuda:0')


x1 = torch.randn(1, 113, 14, 14)

test_inputs = [x1]
