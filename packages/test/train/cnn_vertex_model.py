import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.mesh_based_model import *
from packages.network.vertex_based_model import *


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

        model = HalfEdgeCNNVertexModel(
            in_channel_num=5, mid_channel_num=32, category_num=2, neighbor_type_list=['H', 'H', 'H', 'H']
        )

        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)

        x = half_edges_to_tensor(mesh.half_edges)

        labels = torch.tensor([
            0,
            0,
            0,
            0,
            1
        ])

        # before
        print("Before training:")
        with torch.no_grad():
            outputs = model(x, mesh.half_edges, mesh.vertices)
            probabilities = F.softmax(outputs, dim=1)
            print(probabilities)

        for epoch in range(50):
            optimizer.zero_grad()
            
            outputs = model(x, mesh.half_edges, mesh.vertices)
            
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        # after
        print("After training:")
        with torch.no_grad():
            outputs = model(x, mesh.half_edges, mesh.vertices)
            probabilities = F.softmax(outputs, dim=1)
            print(probabilities)

'''
Before training:
tensor([[0.5063, 0.4937],
        [0.5076, 0.4924],
        [0.5063, 0.4937],
        [0.5076, 0.4924],
        [0.5058, 0.4942]])
Epoch 5, Loss: 0.47981223464012146
Epoch 10, Loss: 0.37422969937324524
Epoch 15, Loss: 0.12304947525262833
Epoch 20, Loss: 0.008748207241296768
Epoch 25, Loss: 0.00024173021665774286
Epoch 30, Loss: 1.6235715520451777e-05
Epoch 35, Loss: 6.675714416815026e-07
Epoch 40, Loss: 1.1920927533992653e-07
Epoch 45, Loss: 9.536741885085576e-08
Epoch 50, Loss: 9.536741885085576e-08
After training:
tensor([[1.0000e+00, 1.4909e-14],
        [1.0000e+00, 2.3577e-07],
        [1.0000e+00, 1.4909e-14],
        [1.0000e+00, 2.3577e-07],
        [6.0020e-09, 1.0000e+00]])
'''