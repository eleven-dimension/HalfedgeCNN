import torch
import torch.nn as nn
import torch.nn.functional as F

from packages.half_edge.neighbor import *
from packages.half_edge.register_neighbor import *


class HalfEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, neighbor_type):
        super(HalfEdgeConv, self).__init__()
        
        self.neighbor_type = neighbor_type

        self.neighbor_func_map = register_neighbor_functions('packages.half_edge.neighbor')
        
        self.neighbor_func = self.neighbor_func_map[self.neighbor_type]
        self.linear_layer = nn.Linear(in_channels * len(self.neighbor_func(HalfEdge(0))), out_channels)


    def forward(self, x, half_edges):
        # x is a [N, in_channels] tensor, where N is the number of nodes.
        # half_edges is a list of HalfEdge objects.
        # print(x)

        # Create an empty tensor to store the output features.
        out = torch.empty((x.shape[0], self.linear_layer.out_features), device=x.device)

        # For each node, aggregate the features of its neighbors.
        for i, he in enumerate(half_edges):
            # Get the features of the node and its neighbors.
            neighbors = self.neighbor_func(he)

            # if i == 0:
            #     print("--------forward--------")
            #     for neighbor in neighbors:
            #         print(neighbor)
            #         print(neighbor.id)

            neighbor_features = torch.cat([x[neighbor.id] for neighbor in neighbors], dim=-1)

            # if i == 0:
            #     print(neighbor_features)
            # Pass them through the linear layer.
            out[i] = self.linear_layer(neighbor_features)

        return F.relu(out)
