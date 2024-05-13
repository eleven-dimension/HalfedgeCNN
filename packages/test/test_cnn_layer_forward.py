from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.layer import *


if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()
        mesh.print_mesh_info()

        halfEdgeCNNLayer = HalfEdgeConv(
            in_channels=3, out_channels=7, neighbor_type='A'
        )

        x = torch.randn((18, 3))

        with torch.no_grad():
            out = halfEdgeCNNLayer(x, mesh.half_edges)

        print(out.shape) # (18, 7)

        