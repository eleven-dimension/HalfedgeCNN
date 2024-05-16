import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.model import *


def half_edges_to_tensor(half_edges):
    features = [torch.from_numpy(he.features) for he in half_edges]
    features_tensor = torch.stack(features)
    return features_tensor


if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()

        model = HalfEdgeCNNModel(
            in_channel_num=5, mid_channel_num=32, pool_output_size=4, category_num=2, neighbor_type_list=['A', 'E', 'H']
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=0.01)

        x = half_edges_to_tensor(mesh.half_edges)
        labels = torch.tensor([0])

        # before
        print("Before training:")
        with torch.no_grad():
            outputs = model(x, mesh.half_edges)
            probabilities = F.softmax(outputs, dim=0)
            print(probabilities)

        for epoch in range(50):
            optimizer.zero_grad()
            
            outputs = model(x, mesh.half_edges).unsqueeze(0)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 1:
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        # after
        print("After training:")
        with torch.no_grad():
            outputs = model(x, mesh.half_edges)
            probabilities = F.softmax(outputs, dim=0)
            print(probabilities)
