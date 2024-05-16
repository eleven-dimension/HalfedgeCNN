import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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