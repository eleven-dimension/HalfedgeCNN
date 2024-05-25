from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *


if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()

        print("------------------------")
        print("opposite angles here:")
        print(mesh.half_edges[1].opposite_angles())
        print(mesh.half_edges[3].opposite_angles())
        print(mesh.half_edges[6].opposite_angles())
        print(mesh.half_edges[2].opposite_angles())