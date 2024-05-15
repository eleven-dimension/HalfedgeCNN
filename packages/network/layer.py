import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib.util
import inspect

from packages.half_edge.neighbor import *


class HalfEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, neighbor_type):
        super(HalfEdgeConv, self).__init__()
        self.neighbor_type = neighbor_type
        self.neighbor_func_map = {}

        self.register_neighbor_functions('packages.half_edge.neighbor')
        
        self.neighbor_func = self.neighbor_func_map[self.neighbor_type]
        self.linear_layer = nn.Linear(in_channels * len(self.neighbor_func(HalfEdge(0))), out_channels)


    def register_neighbor_functions(self, module_name):
        try:
            spec = importlib.util.find_spec(module_name)
            if spec is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and name.startswith('get_neighbors_'):
                        neighbor_type = name.split('_')[-1]
                        self.neighbor_func_map[neighbor_type] = obj
        except ImportError as e:
            print(f"Failed to import module {module_name}: {e}")


    def forward(self, x, half_edges):
        # x is a [N, in_channels] tensor, where N is the number of nodes.
        # half_edges is a list of HalfEdge objects.

        # Create an empty tensor to store the output features.
        out = torch.empty((x.shape[0], self.linear_layer.out_features), device=x.device)

        # For each node, aggregate the features of its neighbors.
        for i, he in enumerate(half_edges):
            # Get the features of the node and its neighbors.
            neighbors = self.neighbor_func(he)
            neighbor_features = torch.cat([x[neighbor.vertex.id] for neighbor in neighbors], dim=-1)

            # Pass them through the linear layer.
            out[i] = self.linear_layer(neighbor_features)

        return F.relu(out)
