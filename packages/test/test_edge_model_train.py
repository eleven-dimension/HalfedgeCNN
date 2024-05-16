import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.mesh_based_model import *
from packages.network.halfedge_based_model import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def half_edges_to_tensor(half_edges):
    features = [torch.from_numpy(he.features) for he in half_edges]
    features_tensor = torch.stack(features)
    return features_tensor


if __name__ == "__main__":
    set_seed(0)

    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()

        model = HalfEdgeCNNEdgeModel(
            in_channel_num=5, mid_channel_num=32, category_num=2, neighbor_type_list=['A', 'E', 'H']
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=0.01)

        x = half_edges_to_tensor(mesh.half_edges)

        labels = torch.tensor([
            0, 0, 0,
            0, 0, 0,
            0, 1, 1,
            0, 1, 1,
            0, 1, 1,
            0, 1, 1
        ])

        # before
        print("Before training:")
        with torch.no_grad():
            outputs = model(x, mesh.half_edges)
            probabilities = F.softmax(outputs, dim=1)
            print(probabilities)

        for epoch in range(50):
            optimizer.zero_grad()
            
            outputs = model(x, mesh.half_edges)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 10 == 1:
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        # after
        print("After training:")
        with torch.no_grad():
            outputs = model(x, mesh.half_edges)
            probabilities = F.softmax(outputs, dim=1)
            print(probabilities)

'''
Before training:
tensor([[0.5000, 0.5000],
        [0.4897, 0.5103],
        [0.5000, 0.5000],
        [0.4897, 0.5103],
        [0.5000, 0.5000],
        [0.5000, 0.5000],
        [0.4953, 0.5047],
        [0.4839, 0.5161],
        [0.5000, 0.5000],
        [0.5000, 0.5000],
        [0.4857, 0.5143],
        [0.5000, 0.5000],
        [0.4953, 0.5047],
        [0.4839, 0.5161],
        [0.5000, 0.5000],
        [0.5000, 0.5000],
        [0.4857, 0.5143],
        [0.5000, 0.5000]])
Epoch 1, Loss: 0.6898590922355652
Epoch 11, Loss: 0.003111132187768817
Epoch 21, Loss: 1.589456815054291e-07
Epoch 31, Loss: 0.0
Epoch 41, Loss: 0.0
After training:
tensor([[1.0000e+00, 7.2168e-13],
        [1.0000e+00, 1.8746e-10],
        [1.0000e+00, 3.6187e-13],
        [1.0000e+00, 1.8746e-10],
        [1.0000e+00, 3.6187e-13],
        [1.0000e+00, 7.2168e-13],
        [1.0000e+00, 1.8321e-13],
        [4.4137e-14, 1.0000e+00],
        [2.5965e-11, 1.0000e+00],
        [1.0000e+00, 3.2894e-13],
        [2.5664e-10, 1.0000e+00],
        [6.6082e-15, 1.0000e+00],
        [1.0000e+00, 1.8321e-13],
        [4.4137e-14, 1.0000e+00],
        [2.5965e-11, 1.0000e+00],
        [1.0000e+00, 3.2894e-13],
        [2.5664e-10, 1.0000e+00],
        [6.6082e-15, 1.0000e+00]])
'''