import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.layer import *


def half_edges_to_tensor(half_edges):
    features = [torch.from_numpy(he.features) for he in half_edges]
    features_tensor = torch.stack(features)
    return features_tensor


if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()

        conv = HalfEdgeConv(
            in_channels=5, out_channels=2, neighbor_type='A'
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(conv.parameters(), lr=0.01)

        x = half_edges_to_tensor(mesh.half_edges)
        labels = torch.tensor([
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0,
            0, 0, 0
        ])

        # before
        print("Before training:")
        with torch.no_grad():
            outputs = conv(x, mesh.half_edges)
            probabilities = F.softmax(outputs, dim=1)
            print(probabilities)

        for epoch in range(120):
            optimizer.zero_grad()
            
            outputs = conv(x, mesh.half_edges)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 1:
                print(f'Epoch {epoch+1}, Loss: {loss.item()}')

        # after
        print("After training:")
        with torch.no_grad():
            outputs = conv(x, mesh.half_edges)
            probabilities = F.softmax(outputs, dim=1)
            print(probabilities)
