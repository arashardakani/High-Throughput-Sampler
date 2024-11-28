import argparse
import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import gc
from .pgates import *


class PIEmbedding(nn.Module):
    def __init__(
        self,
        input_shape: list[int],
        device: str = "cpu",
        batch_size: int = 1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.device = device
        self.batch_size = batch_size

        self.parameters_list = nn.ParameterList()
        for size in input_shape:
            param = nn.Parameter(torch.randn(batch_size, size, device=device))
            self.parameters_list.append(param)

        self.activation = torch.nn.Sigmoid()  

    def forward(self):
        outputs = []
        for param in self.parameters_list:
            param.data.clamp_(-3.5, 3.5)
            output_tensor = self.activation(2 * param)
            outputs.append(output_tensor)
        return outputs




class CircuitModel(nn.Module):
    """Combinational Circuit instantiated from a PySAT CNF problem"""

    def __init__(self, **kwargs):
        # read cnf file
        super().__init__()
        self.pytorch_model = kwargs["pytorch_model"]
        self.num_inputs = kwargs["num_inputs"]
        self.num_outputs = kwargs["num_outputs"]
        exec(self.pytorch_model)
        class_object = locals()['DUT']
        self.emb = PIEmbedding([1] * self.num_inputs, kwargs["device"], kwargs["batch_size"])
        self.probabilistic_circuit_model = class_object(kwargs["batch_size"], kwargs["device"])
        

    def forward(self):
        x = self.emb()
        out, vars = self.probabilistic_circuit_model(x)
        return out, vars

