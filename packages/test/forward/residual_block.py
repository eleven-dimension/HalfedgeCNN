from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.residual_block import *

def half_edges_to_tensor(half_edges):
    features = [torch.from_numpy(he.features) for he in half_edges]
    features_tensor = torch.stack(features)
    return features_tensor

if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()

        block = HalfEdgeResidualBlock(
            channels=5, neighbor_type='H'
        )

        x = half_edges_to_tensor(mesh.half_edges)

        with torch.no_grad():
            out = block(x, mesh.half_edges)

        print(out.shape) # (18, 5)
