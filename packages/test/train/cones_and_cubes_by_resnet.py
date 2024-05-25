import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os

from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.mesh_based_resnet_model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def half_edges_to_tensor(half_edges):
    features = [torch.from_numpy(he.features) for he in half_edges]
    features_tensor = torch.stack(features).to(device)
    return features_tensor


def get_mesh_probabilities(model, all_meshes):
    probabilities = []

    with torch.no_grad():
        for mesh in all_meshes:
            x = half_edges_to_tensor(mesh.half_edges)
            outputs = model(x, mesh.half_edges)
            prob = F.softmax(outputs, dim=0)
            probabilities.append(prob)

    return probabilities


if __name__ == "__main__":
    set_seed(0)

    model = HalfEdgeResNetMeshModel(
        in_channel_num=5, mid_channel_num=64, pool_output_size=16, category_num=2, neighbor_type_list=['H', 'H', 'H', 'H']
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    cone_meshes = []
    cube_meshes = []

    for i in range(1, 5):
        mesh = Mesh()
        filepath = f'./packages/obj/Cone{i}.obj'
        if mesh.load_obj(filepath):
            mesh.convert_obj_format_to_mesh()
            cone_meshes.append(mesh)

    for i in range(1, 5):
        mesh = Mesh()
        filepath = f'./packages/obj/Cube{i}.obj'
        if mesh.load_obj(filepath):
            mesh.convert_obj_format_to_mesh()
            cube_meshes.append(mesh)

    all_meshes = cone_meshes + cube_meshes
    all_labels = torch.tensor([0] * len(cone_meshes) + [1] * len(cube_meshes)).to(device)

    print(f"all labels: {all_labels}")

    # before
    print("Before training:")
    print(get_mesh_probabilities(model, all_meshes))

    for epoch in range(70):
        for mesh, label in zip(all_meshes, all_labels):
            optimizer.zero_grad()

            x = half_edges_to_tensor(mesh.half_edges)
            outputs = model(x, mesh.half_edges).unsqueeze(0)

            # print(outputs)
            # print(label.unsqueeze(0))

            loss = criterion(outputs, label.unsqueeze(0))

            loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')
    
    # after
    print("After training:")
    print(get_mesh_probabilities(model, all_meshes))

    model_path = './packages/model/cone_and_cube_resnet_model.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.to('cpu')

    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")

'''
    all labels: tensor([0, 0, 0, 0, 1, 1, 1, 1], device='cuda:0')
    Before training:
    [tensor([0.4650, 0.5350], device='cuda:0'), tensor([0.4861, 0.5139], device='cuda:0'), tensor([0.4757, 0.5243], device='cuda:0'), tensor([0.4337, 0.5663], device='cuda:0'), tensor([0.4891, 0.5109], device='cuda:0'), tensor([0.4542, 0.5458], device='cuda:0'), tensor([0.4729, 0.5271], device='cuda:0'), tensor([0.4886, 0.5114], device='cuda:0')]
    Epoch 10, Loss: 0.5764193534851074
    Epoch 20, Loss: 0.37149766087532043
    Epoch 30, Loss: 0.03880003094673157
    Epoch 40, Loss: 0.05297886207699776
    Epoch 50, Loss: 0.043909721076488495
    Epoch 60, Loss: 0.0030359390657395124
    Epoch 70, Loss: 0.0007496645557694137
    After training:
    [tensor([1.0000e+00, 7.8276e-24], device='cuda:0'), tensor([0.9903, 0.0097], device='cuda:0'), tensor([1.0000e+00, 7.3916e-12], device='cuda:0'), tensor([1., 0.], device='cuda:0'), tensor([1.0357e-09, 1.0000e+00], device='cuda:0'), tensor([0.0060, 0.9940], device='cuda:0'), tensor([0.0018, 0.9982], device='cuda:0'), tensor([7.3433e-04, 9.9927e-01], device='cuda:0')]
    Model saved to ./packages/model/cone_and_cube_resnet_model.pth
'''