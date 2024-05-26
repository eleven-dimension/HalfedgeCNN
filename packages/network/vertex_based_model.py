import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.half_edge.neighbor import *
from packages.network.conv_layer import *
from packages.network.halfedge_to_vertex_layer import *
from packages.half_edge.register_neighbor import *


class HalfEdgeCNNVertexModel(nn.Module):
    def __init__(self, in_channel_num, mid_channel_num, category_num, neighbor_type_list):
        super(HalfEdgeCNNVertexModel, self).__init__()

        self.neighbor_type_list = neighbor_type_list
        self.convs = nn.ModuleList()
        self.vertex_out = HalfedgeToVertexLayer()

        # Create a HalfEdgeConv layer for each neighbor type.
        for i, neighbor_type in enumerate(self.neighbor_type_list):
            in_channels = in_channel_num if i == 0 else mid_channel_num
            out_channels = category_num if i == len(self.neighbor_type_list) - 1 else mid_channel_num
            self.convs.append(HalfEdgeConv(in_channels, out_channels, neighbor_type))


    def forward(self, x, half_edges, vertices):
        # Apply each HalfEdgeConv layer in sequence.
        for conv in self.convs:
            # print(x.shape)
            x = conv(x, half_edges)

        x = self.vertex_out(x, half_edges, vertices)
        return x
