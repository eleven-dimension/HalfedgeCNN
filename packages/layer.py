import torch
import torch.nn as nn
import torch.nn.functional as F



class HalfEdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HalfEdgeConv, self).__init__()
        self.lin = nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, half_edges):
        # x is a [N, in_channels] tensor, where N is the number of nodes.
        # half_edges is a list of HalfEdge objects.

        # Create an empty tensor to store the output features.
        out = torch.empty_like(x)

        # For each node, aggregate the features of its neighbors.
        for i, he in enumerate(half_edges):
            # Get the features of the node and its neighbors.
            self_feature = x[he.vertex.id]
            next_feature = x[he.next.vertex.id]
            twin_feature = x[he.twin.vertex.id] if he.has_twin() else torch.zeros_like(self_feature)

            # Concatenate the features and pass them through the linear layer.
            neighbor_feature = torch.cat([next_feature, twin_feature], dim=-1)
            out[i] = self.lin(neighbor_feature)

        return F.relu(out)
