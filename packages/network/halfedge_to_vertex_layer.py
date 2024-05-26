import torch
import torch.nn as nn
import torch.nn.functional as F

class HalfedgeToVertexLayer(nn.Module):
    def __init__(self):
        super(HalfedgeToVertexLayer, self).__init__()
        
    
    def forward(self, x, half_edges, vertices):
        # x is a [N, in_channels] tensor, where N is the number of nodes.
        # half_edges is a list of HalfEdge objects.
        # Create an empty tensor to store the output features.
        out = torch.zeros((len(vertices), x.shape[-1]), device=x.device)

        for i, he in enumerate(half_edges):
            from_vertex_id = he.O().vertex.id
            out[from_vertex_id] = torch.add(out[from_vertex_id], x[i])

        for i in range(len(vertices)):
            out[i] = torch.div(out[i], vertices[i].valence)

        return out