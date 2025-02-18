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
        self.fc = torch.nn.Linear(224, 256)

    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + torch.ones(v1.size())
        v3 = torch.nn.functional.relu(v2)
        return v3

class MyDataset(Dataset):

    def __init__(self, samples, targets):
        self.samples = samples
        self.targets = targets

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.targets[index]
        return (sample, target)

class Model(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(224, 256)

    def forward(self, x1):
        v1 = self.fc(x1)
        v2 = v1 + torch.ones(v1.size())
        v3 = torch.nn.functional.relu(v2)
        return v3


func = Model().to('cuda:0')

x1 = 1

test_inputs = [x1]
