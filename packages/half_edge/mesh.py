import numpy as np
from packages.half_edge.half_edge import *

class Mesh:
    def __init__(self):
        self.display_vertices = []
        self.display_faces = []
        self.vertices = []
        self.half_edges = []
        self.faces = []
        self.edges = []

    def add_vertex(self, vertex):
        self.vertices.append(vertex)

    def add_half_edge(self, half_edge):
        self.half_edges.append(half_edge)

    def add_face(self, face):
        self.faces.append(face)

    def add_edge(self, edge):
        self.edges.append(edge)


    def load_obj(self, filepath):
        if len(self.display_vertices) > 0:
            self.display_vertices.clear()
            self.display_faces.clear()

        if not filepath.endswith(".obj"):
            print("Only obj file is supported.")
            return False

        try:
            with open(filepath, 'r') as file:
                for line in file:
                    parts = line.split()
                    if not parts:
                        continue

                    prefix = parts[0]
                    if prefix == "v":
                        vertex = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                        self.display_vertices.append(vertex)
                    elif prefix == "f":
                        face = np.array([int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1])  # OBJ indices start at 1
                        self.display_faces.append(face)
        except IOError:
            print(f"Failed to open file: {filepath}")
            return False

        return True


    def save_obj(self, filepath):
        if not filepath.endswith(".obj"):
            print("Only obj file is supported.")
            return False

        try:
            with open(filepath, 'w') as out_file:
                for vertex in self.display_vertices:
                    out_file.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")

                for face in self.display_faces:
                    out_file.write(f"f {face[0]} {face[1]} {face[2]}\n")
        except IOError as e:
            print(f"Failed to open file: {filepath}")
            print(e)
            return False

        return True


    def convert_obj_format_to_mesh(self):
        if len(self.vertices) > 0:
            self.vertices.clear()
            self.faces.clear()
            self.half_edges.clear()
            self.edges.clear()

        # For tracking edges and linking twins
        edge_map = {}

        # Create vertices
        for i, v in enumerate(self.display_vertices):
            vertex = Vertex(i, v)
            self.vertices.append(vertex)

        # Create half-edges, faces, and vertices
        for i, face_indices in enumerate(self.display_faces):
            face = Face(i)
            face_half_edges = []
            face_vertex_list = []

            for j in range(3):
                start_index = face_indices[j]
                end_index = face_indices[(j + 1) % 3]

                he = HalfEdge(len(self.half_edges))
                he.vertex = self.vertices[end_index]  # pointing to the end vertex
                he.face = face
                face_half_edges.append(he)

                edge_key = tuple(sorted((start_index, end_index)))
                if edge_key not in edge_map:
                    edge_map[edge_key] = he
                else:
                    twin = edge_map[edge_key]
                    he.twin = twin
                    twin.twin = he

                # Linking vertices to their outgoing half-edges if not already linked
                if self.vertices[start_index].he is None:
                    self.vertices[start_index].he = he

                self.half_edges.append(he)

            # Linking half-edges cyclically
            for k in range(3):
                face_half_edges[k].next = face_half_edges[(k + 1) % 3]

            face.he = face_half_edges[0]
            
            for index in face_indices:
                vertex = self.vertices[index]  # assuming self.vertices is a list of Vertex objects
                face_vertex_list.append(vertex)
            face.vertex_list = face_vertex_list
            
            self.faces.append(face)

        # Creating edges based on half-edges
        for he in edge_map.values():
            if he.edge is None:  # Ensure each edge is created once
                edge = Edge(he, len(self.edges))
                self.edges.append(edge)
                he.edge = edge
                if he.twin:
                    he.twin.edge = edge

        # Calculating input features
        for he in self.half_edges:
            he.calc_input_features()

        # Calculating the valence of each vertex
        for he in self.half_edges:
            he.O().vertex.valence += 1


    def convert_mesh_to_obj_format(self):
        if len(self.display_vertices) > 0:
            self.display_vertices.clear()
            self.display_faces.clear()
        
        # Dictionary to keep track of vertex indices
        indices = {}
        for idx, vertex in enumerate(self.vertices):
            indices[vertex] = idx + 1  # OBJ format is 1-based index
            self.display_vertices.append(vertex.pos)

    # Gather face data based on vertex indices
        for face in self.faces:
            if face.vertices():
                face_vert_ids = [indices[vertex] for vertex in face.vertices()]
                self.display_faces.append(face_vert_ids)


    def print_mesh_info(self):
        print("Vertices:", len(self.vertices))
        for vertex in self.vertices:
            print(vertex)
        print("------------")


        print("HalfEdges:", len(self.half_edges))
        for half_edge in self.half_edges:
            print(half_edge)
        print("------------")

        print("Faces:", len(self.faces))
        for face in self.faces:
            print(face)
        print("------------")

        print("Edges:", len(self.edges))
        for edge in self.edges:
            print(edge)
        print("------------")

    
    def print_mesh_halfedges(self):
        print("HalfEdges:", len(self.half_edges))
        for half_edge in self.half_edges:
            print(half_edge)
        print("------------")