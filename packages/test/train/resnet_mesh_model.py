import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.mesh_based_model import *
from packages.network.mesh_based_resnet_model import *

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

        model = HalfEdgeResNetMeshModel(
           in_channel_num=5, mid_channel_num=32, pool_output_size=4, category_num=2, neighbor_type_list=['H', 'H', 'H']
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

'''
Before training:
tensor([0.5054, 0.4946])
Epoch 1, Loss: 0.6824349761009216
Epoch 11, Loss: 0.0
Epoch 21, Loss: 0.0
Epoch 31, Loss: 0.0
Epoch 41, Loss: 0.0
After training:
tensor([1., 0.])
'''