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

        self.features = np.zeros(5, dtype=np.float32)


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


    def dihedral_angle(self):
        v1 = self.vertex.pos - self.O().vertex.pos
        v2 = self.N().vertex.pos - self.vertex.pos
        v3 = self.ON().vertex.pos - self.O().vertex.pos

        normal1 = np.cross(v1, v2)
        normal2 = np.cross(-v1, v3)

        cosine_angle = np.dot(normal1, normal2) / (np.linalg.norm(normal1) * np.linalg.norm(normal2))
        return cosine_angle


    @staticmethod
    def angle_between(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return cosine_angle


    def opposite_angles(self):
        angle1 = HalfEdge.angle_between(self.vertex.pos, self.N().vertex.pos, self.O().vertex.pos)
        angle2 = HalfEdge.angle_between(self.vertex.pos, self.ON().vertex.pos, self.O().vertex.pos)
        return angle1, angle2
    

    @staticmethod
    def height_to_base_ratio(p1, p2, p3):
        v1 = p1 - p2
        v2 = p3 - p2
        area = np.linalg.norm(np.cross(v1, v2))
        base = np.linalg.norm(p3 - p1)
        ratio = area / base ** 2
        return ratio


    def height_to_base_ratios(self):
        ratio1 = HalfEdge.height_to_base_ratio(self.vertex.pos, self.N().vertex.pos, self.O().vertex.pos)
        ratio2 = HalfEdge.height_to_base_ratio(self.vertex.pos, self.ON().vertex.pos, self.O().vertex.pos)
        return ratio1, ratio2
    

    def calc_input_features(self):
        self.features[0] = self.dihedral_angle()
        self.features[1], self.features[2] = self.opposite_angles()
        self.features[3], self.features[4] = self.height_to_base_ratios()
    
    
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
