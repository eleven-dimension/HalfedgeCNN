from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.conv_layer import *
from packages.network.mesh_based_model import *


if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()
        
        model = HalfEdgeCNNMeshModel(
            in_channel_num=5, mid_channel_num=32, pool_output_size=4, category_num=2, neighbor_type_list=['A', 'E', 'H']
        )

        x = torch.randn((18, 5))

        with torch.no_grad():
            out = model(x, mesh.half_edges)

        print(out.shape) # 2

        print(out)
        _, predicted = torch.max(out, 0)
        print(predicted.item())