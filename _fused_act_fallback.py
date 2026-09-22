import torch
from torch import nn
import torch.nn.functional as F

class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale
    def forward(self, input):
        return F.leaky_relu(input + self.bias.reshape(1, -1, 1, 1), self.negative_slope) * self.scale

def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    shape = (1, -1, 1, 1) if input.ndim == 4 else (1, -1)
    return F.leaky_relu(input + bias.reshape(*shape), negative_slope) * scale
