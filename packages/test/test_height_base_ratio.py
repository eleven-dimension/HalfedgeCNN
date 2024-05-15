from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *


if __name__ == "__main__":
    # 0.5
    print(
        HalfEdge.height_to_base_ratio(
            np.array([0, 0, 1]), 
            np.array([0, 0, 0]), 
            np.array([0, 1, 0])
        )
    )