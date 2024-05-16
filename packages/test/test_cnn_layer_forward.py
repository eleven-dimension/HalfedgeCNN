from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.layer import *

def half_edges_to_tensor(half_edges):
    features = [torch.from_numpy(he.features) for he in half_edges]
    features_tensor = torch.stack(features)
    return features_tensor

if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()
        mesh.print_mesh_info()

        halfEdgeCNNLayer = HalfEdgeConv(
            in_channels=5, out_channels=7, neighbor_type='A'
        )

        x = half_edges_to_tensor(mesh.half_edges)

        with torch.no_grad():
            out = halfEdgeCNNLayer(x, mesh.half_edges)

        print(out.shape) # (18, 7)

        