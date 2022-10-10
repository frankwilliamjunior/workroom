import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torch.autograd
import os

class MYSELUImpl(torch.autograd.Function):
    @staticmethod
    def symbolic(g,x,p):
        printf("======================call symbolic")
        return g.op("MYSELU",x,p,
        g.op("Constant",value_t = torch.tensor([3,2,1])))