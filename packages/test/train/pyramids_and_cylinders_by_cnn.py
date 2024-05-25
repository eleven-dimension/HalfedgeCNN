import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score
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
            prob = torch.nn.functional.softmax(outputs, dim=0)
            probabilities.append(prob)

    return probabilities

def load_meshes(directory, label, limit):
    meshes = []
    for i in range(limit):
        mesh = Mesh()
        filepath = os.path.join(directory, f'{label}_{i}.obj')
        if mesh.load_obj(filepath):
            mesh.convert_obj_format_to_mesh()
            meshes.append(mesh)
    return meshes

if __name__ == "__main__":
    set_seed(0)
    
    TRAIN_SIZE_PER_CLASS = 90
    VALIDATION_SIZE_PER_CLASS = 10
    TOTAL_SIZE_PER_CLASS = 100
    
    EPOCH_NUM = 5

    model = HalfEdgeCNNMeshModel(
        in_channel_num=5, mid_channel_num=128, pool_output_size=16, category_num=2, neighbor_type_list=['H', 'H', 'H', 'H']
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Begin loading meshes...")
    # Load datasets
    cylinder_meshes = load_meshes('./packages/obj/dataset/cylinder', 'cylinder', TOTAL_SIZE_PER_CLASS)
    pyramid_meshes = load_meshes('./packages/obj/dataset/pyramid', 'pyramid', TOTAL_SIZE_PER_CLASS)

    # Split into training and validation sets
    train_cylinder_meshes = cylinder_meshes[:TRAIN_SIZE_PER_CLASS]
    val_cylinder_meshes = cylinder_meshes[TRAIN_SIZE_PER_CLASS:TRAIN_SIZE_PER_CLASS + VALIDATION_SIZE_PER_CLASS]
    train_pyramid_meshes = pyramid_meshes[:TRAIN_SIZE_PER_CLASS]
    val_pyramid_meshes = pyramid_meshes[TRAIN_SIZE_PER_CLASS:TRAIN_SIZE_PER_CLASS + VALIDATION_SIZE_PER_CLASS]

    train_meshes = train_cylinder_meshes + train_pyramid_meshes
    val_meshes = val_cylinder_meshes + val_pyramid_meshes

    train_labels = torch.tensor([0] * TRAIN_SIZE_PER_CLASS + [1] * TRAIN_SIZE_PER_CLASS).to(device)
    val_labels = torch.tensor([0] * VALIDATION_SIZE_PER_CLASS + [1] * VALIDATION_SIZE_PER_CLASS).to(device)


    print("Begin training...")
    
    epoch_losses = []
    use_tqdm = True
    
    # Training phase
    for epoch in range(EPOCH_NUM):
        model.train()
        permutation = torch.randperm(len(train_meshes))
        train_meshes = [train_meshes[i] for i in permutation]
        train_labels = train_labels[permutation]
        
        # print(train_labels)

        epoch_loss = 0
        
        if use_tqdm:
            pbar = tqdm(total=len(train_meshes), desc=f'Epoch {epoch + 1}')
        else:
            pbar = None
        
        mesh_cnt = 0
        for mesh, label in zip(train_meshes, train_labels):
            optimizer.zero_grad()

            x = half_edges_to_tensor(mesh.half_edges)
            outputs = model(x, mesh.half_edges).unsqueeze(0)
            # print("-----")
            # print(outputs)
            # print(label.unsqueeze(0))
            loss = criterion(outputs, label.unsqueeze(0))
            # print("loss: ", loss)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            mesh_cnt += 1
            
            if use_tqdm:
                pbar.update(1)
                pbar.set_postfix(loss=epoch_loss / mesh_cnt)
        
        if use_tqdm:
            pbar.close()
            
        epoch_losses.append(epoch_loss / len(train_meshes))
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Loss: {epoch_loss / len(train_meshes)}')

    # Validation phase
    model.eval()
    val_predictions = []
    with torch.no_grad():
        for mesh in val_meshes:
            x = half_edges_to_tensor(mesh.half_edges)
            outputs = model(x, mesh.half_edges)
            _, predicted = torch.max(outputs, 0)
            val_predictions.append(predicted.item())

    val_accuracy = accuracy_score(val_labels.cpu(), val_predictions)
    print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

    model_path = './packages/model/cylinder_and_pyramid_cnn_model.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.to('cpu')

    torch.save(model.state_dict(), model_path)
    
    print(f"Model saved to {model_path}")
    
    # Save the epoch loss data to a file
    loss_path = './packages/model/epoch_losses.txt'
    with open(loss_path, 'w') as f:
        for loss in epoch_losses:
            f.write(f"{loss}\n")
    
    print(f"Loss data saved to {loss_path}")
