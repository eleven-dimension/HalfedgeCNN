import os
import argparse
from packages.generate.generate_meshes import generate_random_shape

def generate_shapes(num_cylinders, num_pyramids):
    # Create directories if they don't exist
    os.makedirs("./packages/obj/dataset/cylinder", exist_ok=True)
    os.makedirs("./packages/obj/dataset/pyramid", exist_ok=True)

    # Generate the specified number of cylinders
    for i in range(num_cylinders):
        file_path = f"./packages/obj/dataset/cylinder/cylinder_{i}.obj"
        generate_random_shape(file_path, 'cylinder')

    # Generate the specified number of pyramids
    for i in range(num_pyramids):
        file_path = f"./packages/obj/dataset/pyramid/pyramid_{i}.obj"
        generate_random_shape(file_path, 'pyramid')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cylinders and pyramids.")
    parser.add_argument('--cylinders', type=int, default=1000, help='Number of cylinders to generate')
    parser.add_argument('--pyramids', type=int, default=1000, help='Number of pyramids to generate')
    args = parser.parse_args()

    generate_shapes(args.cylinders, args.pyramids)
