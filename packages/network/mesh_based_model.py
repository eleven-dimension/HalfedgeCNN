import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.half_edge.neighbor import *
from packages.network.layer import *
from packages.half_edge.register_neighbor import *


class HalfEdgeCNNMeshModel(nn.Module):
    def __init__(self, in_channel_num, mid_channel_num, pool_output_size, category_num, neighbor_type_list):
        super(HalfEdgeCNNMeshModel, self).__init__()

        self.neighbor_type_list = neighbor_type_list
        self.convs = nn.ModuleList()

        # Create a HalfEdgeConv layer for each neighbor type.
        for i, neighbor_type in enumerate(self.neighbor_type_list):
            in_channels = in_channel_num if i == 0 else mid_channel_num
            self.convs.append(HalfEdgeConv(in_channels, mid_channel_num, neighbor_type))

        # Adaptive pooling layer.
        self.pool = nn.AdaptiveAvgPool1d(pool_output_size)

        # The final fully connected layer.
        self.fc = nn.Linear(pool_output_size * mid_channel_num, category_num)


    def forward(self, x, half_edges):
        # Apply each HalfEdgeConv layer in sequence.
        for conv in self.convs:
            # print(x.shape)
            x = conv(x, half_edges)

        # Apply the adaptive pooling layer.
        x = self.pool(x.transpose(0,1)).transpose(0,1)
        # print(x.shape)

        # Flatten the tensor.
        x = x.reshape(-1)
        # print(x.shape)

        # Apply the final fully connected layer.
        out = self.fc(x)
        # print(x.shape)

        return out
