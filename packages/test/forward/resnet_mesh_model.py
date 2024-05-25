from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.conv_layer import *
from packages.network.mesh_based_resnet_model import *


if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()

        model = HalfEdgeResNetMeshModel(
            in_channel_num=5, mid_channel_num=32, pool_output_size=4, category_num=2, neighbor_type_list=['H', 'H', 'H', 'H']
        )

        x = torch.randn((18, 5))

        with torch.no_grad():
            out = model(x, mesh.half_edges)

        print(out.shape) # 2
