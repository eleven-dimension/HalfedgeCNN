# HalfedgeCNN

CG assignment. I don’t know why such a CV-related paper would appear in my CG assignment choice list.

## Getting Started

### Prerequisites
As is shown in `requirements.txt`.

### Usage
#### Binary Classification for Pyramids and Cylinders
2000 data points per class, 1800 for training and 200 for validation
```sh
python -m packages.test.train.pyramids_and_cylinders_by_cnn
```
#### Simple Edge-based Model Test
To mark side edges of a pyramid.
```sh
python -m packages.test.train.cnn_edge_model
```
#### Simple Vertex-based Model Test
To mark the apex of a pyramid.
```sh
python -m packages.test.train.cnn_vertex_model
```
#### Simple Mesh-based Model Test
4 cones and 4 cubes
```sh
python -m packages.test.train.cones_and_cubes_by_cnn
```
#### Simple Mesh-based Resnet-like Model Test
```sh
python -m packages.test.train.resnet_mesh_model
```