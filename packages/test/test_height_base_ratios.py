from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *


if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()
        mesh.print_mesh_info()

        print("------------------------")
        print("opposite angles here:")
        print(mesh.half_edges[1].height_to_base_ratios())
        print(mesh.half_edges[3].height_to_base_ratios())
        print(mesh.half_edges[6].height_to_base_ratios())
        print(mesh.half_edges[2].height_to_base_ratios())