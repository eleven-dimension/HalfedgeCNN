from packages.half_edge.neighbor import *
from packages.half_edge.mesh import *


if __name__ == "__main__":
    # cosine angle = 0
    print(
        HalfEdge.angle_between(
            np.array([0, 0, 1]), 
            np.array([0, 0, 0]), 
            np.array([0, 1, 0])
        )
    )