from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *


if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()

        print("dihedral angle here:")
        print(mesh.half_edges[1].dihedral_angle())
        print(mesh.half_edges[3].dihedral_angle())
        print(mesh.half_edges[6].dihedral_angle())
        print(mesh.half_edges[2].dihedral_angle())