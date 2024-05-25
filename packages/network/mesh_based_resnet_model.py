import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.half_edge.neighbor import *
from packages.network.conv_layer import *
from packages.network.residual_block import *
from packages.half_edge.register_neighbor import *


class HalfEdgeResNetMeshModel(nn.Module):
    def __init__(self, in_channel_num, mid_channel_num, pool_output_size, category_num, neighbor_type_list):
        super(HalfEdgeResNetMeshModel, self).__init__()

        self.neighbor_type_list = neighbor_type_list
        self.convs = nn.ModuleList()
        self.blocks = nn.ModuleList()

        # Create the first HalfEdgeConv layer to lift the channels to mid_channel_num.
        self.convs.append(HalfEdgeConv(in_channel_num, mid_channel_num, neighbor_type_list[0]))

        # Create HalfEdgeResidualBlocks for each neighbor type starting from the second one.
        for neighbor_type in self.neighbor_type_list[1:]:
            self.blocks.append(HalfEdgeResidualBlock(mid_channel_num, neighbor_type))

        # Adaptive pooling layer.
        self.pool = nn.AdaptiveAvgPool1d(pool_output_size)

        # The final fully connected layer.
        self.fc = nn.Linear(pool_output_size * mid_channel_num, category_num)


    def forward(self, x, half_edges):
        # Apply the first HalfEdgeConv layer.
        x = self.convs[0](x, half_edges)

        # Apply each HalfEdgeResidualBlock in sequence.
        for block in self.blocks:
            x = block(x, half_edges)

        # Apply the adaptive pooling layer.
        x = self.pool(x.transpose(0, 1)).transpose(0, 1)

        # Flatten the tensor.
        x = x.reshape(-1)

        # Apply the final fully connected layer.
        out = self.fc(x)

        return out