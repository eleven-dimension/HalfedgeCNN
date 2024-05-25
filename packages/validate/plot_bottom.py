import matplotlib.pyplot as plt

def read_obj_file(file_path):
    vertices = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                if z == 0.0:
                    vertices.append((x, y, z))
    return vertices

def plot_vertices(vertices):
    x_vals = [vertex[0] for vertex in vertices]
    y_vals = [vertex[1] for vertex in vertices]

    plt.figure(figsize=(10, 10))
    plt.scatter(x_vals, y_vals, color='blue', marker='+', s=100)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Vertices with Z = 0')
    plt.show()


if __name__ == "__main__":
    file_path = './packages/obj/dataset/pyramid_0.obj'
    vertices_z0 = read_obj_file(file_path)
    plot_vertices(vertices_z0)
