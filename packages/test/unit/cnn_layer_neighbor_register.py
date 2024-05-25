from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *
from packages.network.conv_layer import *

if __name__ == "__main__":
    conv = HalfEdgeConv(in_channels=5, out_channels=16, neighbor_type='A')
    print(conv.neighbor_func_map)
    print(conv.linear_layer)