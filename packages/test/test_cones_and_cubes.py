import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import os

from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.mesh_based_model import *

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

    model = HalfEdgeCNNMeshModel(
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

    for epoch in range(100):
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

    model_path = './packages/model/cone_and_cube_model.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.to('cpu')

    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")

'''
    all labels: tensor([0, 0, 0, 0, 1, 1, 1, 1], device='cuda:0')
    Before training:
    [tensor([0.4960, 0.5040], device='cuda:0'), tensor([0.5016, 0.4984], device='cuda:0'), tensor([0.4979, 0.5021], device='cuda:0'), tensor([0.4933, 0.5067], device='cuda:0'), tensor([0.4875, 0.5125], device='cuda:0'), tensor([0.5018, 0.4982], device='cuda:0'), tensor([0.4984, 0.5016], device='cuda:0'), tensor([0.4937, 0.5063], device='cuda:0')]
    Epoch 10, Loss: 0.6954051852226257
    Epoch 20, Loss: 0.43153488636016846
    Epoch 30, Loss: 0.282850056886673
    Epoch 40, Loss: 0.19569586217403412
    Epoch 50, Loss: 0.14265751838684082
    Epoch 60, Loss: 0.10856987535953522
    Epoch 70, Loss: 0.08549964427947998
    Epoch 80, Loss: 0.06918029487133026
    Epoch 90, Loss: 0.05720406398177147
    Epoch 100, Loss: 0.04814288765192032
    After training:
    [tensor([1.0000e+00, 3.6336e-15], device='cuda:0'), tensor([1.0000e+00, 7.0521e-13], device='cuda:0'), tensor([1.0000e+00, 6.7302e-14], device='cuda:0'), tensor([1.0000e+00, 2.8104e-13], device='cuda:0'), tensor([1.6427e-13, 1.0000e+00], device='cuda:0'), tensor([0.0469, 0.9531], device='cuda:0'), tensor([0.0469, 0.9531], device='cuda:0'), tensor([0.0469, 0.9531], device='cuda:0')]
'''