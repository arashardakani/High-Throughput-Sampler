import torch

def AND(*args):
    return torch.prod(torch.cat(args, dim = -1), dim = -1).unsqueeze(-1)

def NAND(*args):
    return 1 - torch.prod(torch.cat(args, dim = -1), dim = -1).unsqueeze(-1)

def OR(*args):
    return 1 - torch.prod(1 - torch.cat(args, dim = -1), dim = -1).unsqueeze(-1)

def NOR(*args):
    return torch.prod(1 - torch.cat(args, dim = -1), dim = -1).unsqueeze(-1)

def XOR(a, b):
    return a * (1 - b) + b * (1 - a)

def XNOR(a, b):
    return (1 - a) * (1 - b) + a * b

def NOT(a):
    return 1 - a

def BUF(a):
    return a