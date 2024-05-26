from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.conv_layer import *
from packages.network.vertex_based_model import *


if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()
        
        model = HalfEdgeCNNVertexModel(
            in_channel_num=5, mid_channel_num=32, category_num=2, neighbor_type_list=['H', 'H', 'H']
        )

        x = torch.randn((18, 5))

        with torch.no_grad():
            out = model(x, mesh.half_edges, mesh.vertices)

        print(out.shape) # [5, 2]

