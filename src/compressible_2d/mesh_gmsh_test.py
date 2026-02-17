from compressible_2d.mesh_gmsh import Mesh2D


def test_mesh2d_basic():
    nodes = [
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ]
    cells = [
        [0, 1, 2],
        [0, 2, 3],
    ]
    boundary_edges = [
        (0, 1, 1),
        (1, 2, 2),
        (2, 3, 3),
        (3, 0, 4),
    ]

    mesh = Mesh2D.from_cells(nodes, cells, boundary_edges)
    assert mesh.cell_centroids.shape[0] == 2
    assert mesh.face_nodes.shape[0] >= 4
    assert (mesh.boundary_tags >= -1).all()
