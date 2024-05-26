import numpy as np
from scipy.spatial import ConvexHull, Delaunay

MIN_BASE_POINTS = 3
MAX_BASE_POINTS = 30
BASE_AREA_SIZE = 10
MIN_APEX_HEIGHT = 1
MAX_APEX_HEIGHT = 15
VALID_SHAPE_AREA_THRESHOLD = 3

def is_shape_valid(points):
    # Calculate the convex hull of the points
    hull = ConvexHull(points)
    # Calculate the area of the convex hull
    area = hull.volume
    # Check if the area is too small (indicating points are nearly collinear)
    return area > VALID_SHAPE_AREA_THRESHOLD


def generate_random_rectangle(square_max):
    x1 = np.random.uniform(0, square_max)
    y1 = np.random.uniform(0, square_max)

    max_width = square_max - x1
    max_height = square_max - y1
    
    width = np.random.uniform(0, max_width)
    height = np.random.uniform(0, max_height)

    points = np.array([
        [x1, y1],
        [x1 + width, y1],
        [x1 + width, y1 + height],
        [x1, y1 + height]
    ])

    return points


def generate_shape(base_points, shape_type='pyramid'):
    vertices = []
    faces = []
    
    # Add base vertices
    for p in base_points:
        vertices.append(f"v {p[0]} {p[1]} 0")
    
    if shape_type == 'pyramid':
        apex_z = np.random.uniform(MIN_APEX_HEIGHT, MAX_APEX_HEIGHT)
        apex = [np.mean(base_points[:, 0]), np.mean(base_points[:, 1]), apex_z]
        vertices.append(f"v {apex[0]} {apex[1]} {apex[2]}")
        apex_index = len(vertices)
    
    elif shape_type == 'cylinder':
        height = np.random.uniform(MIN_APEX_HEIGHT, MAX_APEX_HEIGHT)
        # Add top vertices
        for p in base_points:
            vertices.append(f"v {p[0]} {p[1]} {height}")
    
    # Create base faces using Delaunay triangulation
    hull = ConvexHull(base_points)
    delaunay = Delaunay(base_points[hull.vertices])
    for simplex in delaunay.simplices:
        p1 = np.append(base_points[hull.vertices[simplex[0]]], 0)
        p2 = np.append(base_points[hull.vertices[simplex[1]]], 0)
        p3 = np.append(base_points[hull.vertices[simplex[2]]], 0)
        
        # Calculate the normal vector of the triangle
        normal = np.cross(p2 - p1, p3 - p1)
        if normal[2] < 0:
            faces.append(f"f {hull.vertices[simplex[0]] + 1} {hull.vertices[simplex[1]] + 1} {hull.vertices[simplex[2]] + 1}")
        else:
            faces.append(f"f {hull.vertices[simplex[0]] + 1} {hull.vertices[simplex[2]] + 1} {hull.vertices[simplex[1]] + 1}")

        if shape_type == 'cylinder':
            if normal[2] < 0:
                faces.append(f"f {hull.vertices[simplex[0]] + 1 + len(base_points)} {hull.vertices[simplex[2]] + 1 + len(base_points)} {hull.vertices[simplex[1]] + 1 + len(base_points)}")
            else:
                faces.append(f"f {hull.vertices[simplex[0]] + 1 + len(base_points)} {hull.vertices[simplex[1]] + 1 + len(base_points)} {hull.vertices[simplex[2]] + 1 + len(base_points)}")

    # Create side faces
    for i in range(len(hull.vertices)):
        next_i = (i + 1) % len(hull.vertices)
        if shape_type == 'pyramid':
            faces.append(f"f {hull.vertices[i] + 1} {hull.vertices[next_i] + 1} {apex_index}")
        elif shape_type == 'cylinder':
            bottom_i = hull.vertices[i] + 1
            bottom_next_i = hull.vertices[next_i] + 1
            top_i = bottom_i + len(base_points)
            top_next_i = bottom_next_i + len(base_points)
            faces.append(f"f {bottom_i} {bottom_next_i} {top_i}")
            faces.append(f"f {bottom_next_i} {top_next_i} {top_i}")

    return vertices, faces

def write_obj(filename, vertices, faces):
    with open(filename, 'w') as f:
        for v in vertices:
            f.write(v + '\n')
        for face in faces:
            f.write(face + '\n')

def generate_random_shape(filename, shape_type='pyramid'):
    while True:
        if np.random.rand() <= 1/3:
            base_points = generate_random_rectangle(BASE_AREA_SIZE)
            print("rectangle")
        else:
            num_base_points = np.random.randint(MIN_BASE_POINTS, MAX_BASE_POINTS + 1)
            base_points = np.random.rand(num_base_points, 2) * BASE_AREA_SIZE
        if is_shape_valid(base_points):
            break
    vertices, faces = generate_shape(base_points, shape_type)
    write_obj(filename, vertices, faces)

