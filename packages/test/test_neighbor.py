from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *


if __name__ == "__main__":
    mesh = Mesh()
    filepath = './packages/obj/pyramid.obj'

    if mesh.load_obj(filepath):
        mesh.convert_obj_format_to_mesh()
        mesh.print_mesh_info()

        print("------------------------")
        print("heighbor test here:")
        neighbors = get_neighbors_type_A(mesh.half_edges[3])
        for neighbor in neighbors:
            print(neighbor)

        print()
        neighbors = get_neighbors_type_B(mesh.half_edges[3])
        for neighbor in neighbors:
            print(neighbor)

        print()
        neighbors = get_neighbors_type_C(mesh.half_edges[3])
        for neighbor in neighbors:
            print(neighbor)

        print()
        neighbors = get_neighbors_type_D(mesh.half_edges[3])
        for neighbor in neighbors:
            print(neighbor)

        print()
        neighbors = get_neighbors_type_E(mesh.half_edges[3])
        for neighbor in neighbors:
            print(neighbor)

        print()
        neighbors = get_neighbors_type_F(mesh.half_edges[3])
        for neighbor in neighbors:
            print(neighbor)

        print()
        neighbors = get_neighbors_type_G(mesh.half_edges[3])
        for neighbor in neighbors:
            print(neighbor)

        print()
        neighbors = get_neighbors_type_H(mesh.half_edges[3])
        for neighbor in neighbors:
            print(neighbor)

    else:
        print("Failed to load OBJ file.")