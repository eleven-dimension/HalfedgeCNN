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

    model_path = './packages/model/cone_and_cube_cnn_model.pth'
    model.load_state_dict(torch.load(model_path))

    mesh = Mesh()
    filepath = './packages/obj/Validate_Cube_1.obj'
    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()

        x = half_edges_to_tensor(mesh.half_edges)

        with torch.no_grad():
            out = model(x, mesh.half_edges)
            prob = F.softmax(out, dim=0)
            print(prob) # tensor([0.0469, 0.9531], device='cuda:0')
