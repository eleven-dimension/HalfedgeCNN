from typing import List
from packages.half_edge.half_edge import HalfEdge

def get_neighbors_type_A(half_edge: HalfEdge) -> List[HalfEdge]:
    return [half_edge.N(), half_edge.O()]


def get_neighbors_type_B(half_edge: HalfEdge) -> List[HalfEdge]:
    return [half_edge.O(), half_edge.N(), half_edge.P()]


def get_neighbors_type_C(half_edge: HalfEdge) -> List[HalfEdge]:
    return [half_edge.O(), half_edge.N(), half_edge.ON()]


def get_neighbors_type_D(half_edge: HalfEdge) -> List[HalfEdge]:
    return [half_edge.PO(), half_edge.NO(), half_edge.ON(), half_edge.OP()]


def get_neighbors_type_E(half_edge: HalfEdge) -> List[HalfEdge]:
    return [half_edge.PO(), half_edge.NO(), half_edge.N(), half_edge.P(), half_edge.O()]


def get_neighbors_type_F(half_edge: HalfEdge) -> List[HalfEdge]:
    return [half_edge.P(), half_edge.N(), half_edge.ON(), half_edge.OP(), half_edge.O()]


def get_neighbors_type_G(half_edge: HalfEdge) -> List[HalfEdge]:
    return [half_edge.P(), half_edge.N(), half_edge.PO(), half_edge.NO(), half_edge.ON(), half_edge.OP(), half_edge.O()]


def get_neighbors_type_H(half_edge: HalfEdge) -> List[HalfEdge]:
    return [
        half_edge.P(), half_edge.N(), 
        half_edge.PO(), half_edge.NO(), half_edge.ON(), half_edge.OP(), half_edge.O(), 
        half_edge.OPO(), half_edge.ONO()
    ]