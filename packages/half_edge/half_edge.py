import numpy as np

class HalfEdge:
    def __init__(self, _id):
        self.id = _id
        self.exists = True
        self.next = None
        self.twin = None
        self.face = None
        self.vertex = None
        self.edge = None


    def N(self):
        return self.next

    def P(self):
        return self.next.next if self.next else None

    def O(self):
        return self.twin

    def NO(self):
        return self.next.twin if self.next else None

    def PO(self):
        return self.next.next.twin if self.next and self.next.next else None

    def OP(self):
        return self.twin.next.next if self.twin and self.twin.next else None

    def ON(self):
        return self.twin.next if self.twin else None
    
    def ONO(self):
        return self.twin.next.twin if self.twin and self.twin.next else None

    def OPO(self):
        return self.twin.next.next.twin if self.twin and self.twin.next and self.twin.next.next else None

    
    def __str__(self):
        twin_id = self.twin.id if self.twin else None
        next_id = self.next.id if self.next else None
        face_id = self.face.id if self.face else None
        vertex_id = self.vertex.id if self.vertex else None
        edge_id = self.edge.id if self.edge else None
        return f"HalfEdge {self.id} [{self.twin.vertex.id} - {self.vertex.id}]: twin={twin_id}, next={next_id}, face={face_id}, vertex={vertex_id}, edge={edge_id}"
        

class Vertex:
    def __init__(self, _id, pos=None):
        if pos is None:
            pos = np.zeros(3)
        else:
            self.pos = pos
        self.id = _id
        self.exists = True
        self.he = None

    
    def __str__(self):
        he_id = self.he.id if self.he else None
        return f"Vertex {self.id}: exists={self.exists}, position={self.pos}, half_edge={he_id}"



class Face:
    def __init__(self, _id):
        self.he = None
        self.exists = True
        self.id = _id
        self.vertex_list = []

    def vertices(self):
        return self.vertex_list
    
    def __str__(self):
        he_id = self.he.id if self.he else None
        return f"Face {self.id}: exists={self.exists}, half_edge={he_id}, vertices={self.vertices()}"


class Edge:
    def __init__(self, _he, _id):
        self.he = _he
        self.exists = True
        self.id = _id

    def __str__(self):
        he_id = self.he.id if self.he else None
        return f"Edge {self.id}: exists={self.exists}, half_edge={he_id}"