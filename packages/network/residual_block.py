import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.half_edge.neighbor import *
from packages.network.conv_layer import *
from packages.half_edge.register_neighbor import *

class HalfEdgeResidualBlock(nn.Module):
    def __init__(self, channels, neighbor_type):
        super(HalfEdgeResidualBlock, self).__init__()
        self.conv1 = HalfEdgeConv(channels, channels, neighbor_type)
        self.conv2 = HalfEdgeConv(channels, channels, neighbor_type)

    def forward(self, x, half_edges):
        # Save the input for the residual connection
        residual = x

        # Apply the first HalfEdgeConv layer
        out = self.conv1(x, half_edges)
        out = F.relu(out)

        # Apply the second HalfEdgeConv layer
        out = self.conv2(out, half_edges)

        # Add the residual connection
        out = torch.add(out, residual)
        out = F.relu(out)

        return out