from scipy.spatial import Delaunay
from itertools import combinations
import numpy as np
import trimesh
from scipy import ndimage
from skimage.measure import block_reduce


class MultiGridExtractor(object):
    def __init__(self, resolution0, threshold):
        # Attributes
        self.resolution = resolution0
        self.threshold = threshold

        # Voxels are active or inactive,
        # values live on the space between voxels and are either
        # known exactly or guessed by interpolation (unknown)
        shape_voxels = (resolution0,) * 3
        shape_values = (resolution0 + 1,) * 3
        self.values = np.empty(shape_values)
        self.value_known = np.full(shape_values, False)
        self.voxel_active = np.full(shape_voxels, True)

    def query(self):
        # Query locations in grid that are active but unkown
        idx1, idx2, idx3 = np.where(
            ~self.value_known & self.value_active
        )
        points = np.stack([idx1, idx2, idx3], axis=-1)
        return points

    def update(self, points, values):
        # Update locations and set known status to true
        idx0, idx1, idx2 = points.transpose()
        self.values[idx0, idx1, idx2] = values
        self.value_known[idx0, idx1, idx2] = True

        # Update activity status of voxels accordings to new values
        self.voxel_active = ~self.voxel_empty
        # (
        #     # self.voxel_active &
        #     self.voxel_known & ~self.voxel_empty
        # )

    def increase_resolution(self):
        self.resolution = 2 * self.resolution
        shape_values = (self.resolution + 1,) * 3

        value_known = np.full(shape_values, False)
        value_known[::2, ::2, ::2] = self.value_known
        values = upsample3d_nn(self.values)
        values = values[:-1, :-1, :-1]

        self.values = values
        self.value_known = value_known
        self.voxel_active = upsample3d_nn(self.voxel_active)

    @property
    def occupancies(self):
        return (self.values < self.threshold)

    @property
    def value_active(self):
        value_active = np.full(self.values.shape, False)
        # Active if adjacent to active voxel
        value_active[:-1, :-1, :-1] |= self.voxel_active
        value_active[:-1, :-1, 1:] |= self.voxel_active
        value_active[:-1, 1:, :-1] |= self.voxel_active
        value_active[:-1, 1:, 1:] |= self.voxel_active
        value_active[1:, :-1, :-1] |= self.voxel_active
        value_active[1:, :-1, 1:] |= self.voxel_active
        value_active[1:, 1:, :-1] |= self.voxel_active
        value_active[1:, 1:, 1:] |= self.voxel_active

        return value_active

    @property
    def voxel_known(self):
        value_known = self.value_known
        voxel_known =check_voxel_occupied(value_known)
        return voxel_known

    @property
    def voxel_empty(self):
        occ = self.occupancies
        return ~voxels.check_voxel_boundary(occ)


def upsample3d_nn(x):
    xshape = x.shape
    yshape = (2*xshape[0], 2*xshape[1], 2*xshape[2])

    y = np.zeros(yshape, dtype=x.dtype)
    y[::2, ::2, ::2] = x
    y[::2, ::2, 1::2] = x
    y[::2, 1::2, ::2] = x
    y[::2, 1::2, 1::2] = x
    y[1::2, ::2, ::2] = x
    y[1::2, ::2, 1::2] = x
    y[1::2, 1::2, ::2] = x
    y[1::2, 1::2, 1::2] = x

    return y
def check_voxel_occupied(occupancy_grid):
    occ = occupancy_grid

    occupied = (
        occ[..., :-1, :-1, :-1]
        & occ[..., :-1, :-1, 1:]
        & occ[..., :-1, 1:, :-1]
        & occ[..., :-1, 1:, 1:]
        & occ[..., 1:, :-1, :-1]
        & occ[..., 1:, :-1, 1:]
        & occ[..., 1:, 1:, :-1]
        & occ[..., 1:, 1:, 1:]
    )
    return occupied


def check_voxel_boundary(occupancy_grid):
    occupied = check_voxel_occupied(occupancy_grid)
    unoccupied = check_voxel_unoccupied(occupancy_grid)
    return ~occupied & ~unoccupied


class DelauneyMeshExtractor(object):
    """Algorithm for extacting meshes from implicit function using
    delauney triangulation and random sampling."""
    def __init__(self, points, values, threshold=0.):
        self.points = points
        self.values = values
        self.delaunay = Delaunay(self.points)
        self.threshold = threshold

    def update(self, points, values, reduce_to_active=True):
        # Find all active points
        if reduce_to_active:
            active_simplices = self.active_simplices()
            active_point_idx = np.unique(active_simplices.flatten())
            self.points = self.points[active_point_idx]
            self.values = self.values[active_point_idx]

        self.points = np.concatenate([self.points, points], axis=0)
        self.values = np.concatenate([self.values, values], axis=0)
        self.delaunay = Delaunay(self.points)

    def extract_mesh(self):
        threshold = self.threshold
        vertices = []
        triangles = []
        vertex_dict = dict()

        active_simplices = self.active_simplices()
        active_simplices.sort(axis=1)
        for simplex in active_simplices:
            new_vertices = []
            for i1, i2 in combinations(simplex, 2):
                assert(i1 < i2)
                v1 = self.values[i1]
                v2 = self.values[i2]
                if (v1 < threshold) ^ (v2 < threshold):
                    # Subdivide edge
                    vertex_idx = vertex_dict.get((i1, i2), len(vertices))
                    vertex_idx = len(vertices)
                    if vertex_idx == len(vertices):
                        tau = (threshold - v1) / (v2 - v1)
                        assert(0 <= tau <= 1)
                        p = (1 - tau) * self.points[i1] + tau * self.points[i2]
                        vertices.append(p)
                        vertex_dict[i1, i2] = vertex_idx
                    new_vertices.append(vertex_idx)

            assert(len(new_vertices) in (3, 4))
            p0 = self.points[simplex[0]]
            v0 = self.values[simplex[0]]
            if len(new_vertices) == 3:
                i1, i2, i3 = new_vertices
                p1, p2, p3 = vertices[i1], vertices[i2], vertices[i3]
                vol = get_tetrahedon_volume(np.asarray([p0, p1, p2, p3]))
                if vol * (v0 - threshold) <= 0:
                    triangles.append((i1, i2, i3))
                else:
                    triangles.append((i1, i3, i2))
            elif len(new_vertices) == 4:
                i1, i2, i3, i4 = new_vertices
                p1, p2, p3, p4 = \
                    vertices[i1], vertices[i2], vertices[i3], vertices[i4]
                vol = get_tetrahedon_volume(np.asarray([p0, p1, p2, p3]))
                if vol * (v0 - threshold) <= 0:
                    triangles.append((i1, i2, i3))
                else:
                    triangles.append((i1, i3, i2))

                vol = get_tetrahedon_volume(np.asarray([p0, p2, p3, p4]))
                if vol * (v0 - threshold) <= 0:
                    triangles.append((i2, i3, i4))
                else:
                    triangles.append((i2, i4, i3))

        vertices = np.asarray(vertices, dtype=np.float32)
        triangles = np.asarray(triangles, dtype=np.int32)

        return vertices, triangles

    def query(self, size):
        active_simplices = self.active_simplices()
        active_simplices_points = self.points[active_simplices]
        new_points = sample_tetraheda(active_simplices_points, size=size)
        return new_points

    def active_simplices(self):
        occ = (self.values >= self.threshold)
        simplices = self.delaunay.simplices
        simplices_occ = occ[simplices]

        active = (
            np.any(simplices_occ, axis=1) & np.any(~simplices_occ, axis=1)
        )

        simplices = self.delaunay.simplices[active]
        return simplices


def sample_tetraheda(tetraheda_points, size):
    N_tetraheda = tetraheda_points.shape[0]
    volume = np.abs(get_tetrahedon_volume(tetraheda_points))
    probs = volume / volume.sum()

    tetraheda_rnd = np.random.choice(range(N_tetraheda), p=probs, size=size)
    tetraheda_rnd_points = tetraheda_points[tetraheda_rnd]
    weights_rnd = np.random.dirichlet([1, 1, 1, 1], size=size)
    weights_rnd = weights_rnd.reshape(size, 4, 1)
    points_rnd = (weights_rnd * tetraheda_rnd_points).sum(axis=1)
    # points_rnd = tetraheda_rnd_points.mean(1)

    return points_rnd


def get_tetrahedon_volume(points):
    vectors = points[..., :3, :] - points[..., 3:, :]
    volume = 1/6 * np.linalg.det(vectors)
    return volume


class VoxelGrid:
    def __init__(self, data, loc=(0., 0., 0.), scale=1):
        assert(data.shape[0] == data.shape[1] == data.shape[2])
        data = np.asarray(data, dtype=np.bool)
        loc = np.asarray(loc)
        self.data = data
        self.loc = loc
        self.scale = scale
    def to_mesh(self):
        # Shorthand
        occ = self.data

        # Shape of voxel grid
        nx, ny, nz = occ.shape
        # Shape of corresponding occupancy grid
        grid_shape = (nx + 1, ny + 1, nz + 1)

        # Convert values to occupancies
        occ = np.pad(occ, 1, 'constant')

        # Determine if face present
        f1_r = (occ[:-1, 1:-1, 1:-1] & ~occ[1:, 1:-1, 1:-1])
        f2_r = (occ[1:-1, :-1, 1:-1] & ~occ[1:-1, 1:, 1:-1])
        f3_r = (occ[1:-1, 1:-1, :-1] & ~occ[1:-1, 1:-1, 1:])

        f1_l = (~occ[:-1, 1:-1, 1:-1] & occ[1:, 1:-1, 1:-1])
        f2_l = (~occ[1:-1, :-1, 1:-1] & occ[1:-1, 1:, 1:-1])
        f3_l = (~occ[1:-1, 1:-1, :-1] & occ[1:-1, 1:-1, 1:])

        f1 = f1_r | f1_l
        f2 = f2_r | f2_l
        f3 = f3_r | f3_l

        assert(f1.shape == (nx + 1, ny, nz))
        assert(f2.shape == (nx, ny + 1, nz))
        assert(f3.shape == (nx, ny, nz + 1))

        # Determine if vertex present
        v = np.full(grid_shape, False)

        v[:, :-1, :-1] |= f1
        v[:, :-1, 1:] |= f1
        v[:, 1:, :-1] |= f1
        v[:, 1:, 1:] |= f1

        v[:-1, :, :-1] |= f2
        v[:-1, :, 1:] |= f2
        v[1:, :, :-1] |= f2
        v[1:, :, 1:] |= f2

        v[:-1, :-1, :] |= f3
        v[:-1, 1:, :] |= f3
        v[1:, :-1, :] |= f3
        v[1:, 1:, :] |= f3

        # Calculate indices for vertices
        n_vertices = v.sum()
        v_idx = np.full(grid_shape, -1)
        v_idx[v] = np.arange(n_vertices)

        # Vertices
        v_x, v_y, v_z = np.where(v)
        v_x = v_x / nx - 0.5
        v_y = v_y / ny - 0.5
        v_z = v_z / nz - 0.5
        vertices = np.stack([v_x, v_y, v_z], axis=1)

        # Face indices
        f1_l_x, f1_l_y, f1_l_z = np.where(f1_l)
        f2_l_x, f2_l_y, f2_l_z = np.where(f2_l)
        f3_l_x, f3_l_y, f3_l_z = np.where(f3_l)

        f1_r_x, f1_r_y, f1_r_z = np.where(f1_r)
        f2_r_x, f2_r_y, f2_r_z = np.where(f2_r)
        f3_r_x, f3_r_y, f3_r_z = np.where(f3_r)

        faces_1_l = np.stack([
            v_idx[f1_l_x, f1_l_y, f1_l_z],
            v_idx[f1_l_x, f1_l_y, f1_l_z + 1],
            v_idx[f1_l_x, f1_l_y + 1, f1_l_z + 1],
            v_idx[f1_l_x, f1_l_y + 1, f1_l_z],
        ], axis=1)

        faces_1_r = np.stack([
            v_idx[f1_r_x, f1_r_y, f1_r_z],
            v_idx[f1_r_x, f1_r_y + 1, f1_r_z],
            v_idx[f1_r_x, f1_r_y + 1, f1_r_z + 1],
            v_idx[f1_r_x, f1_r_y, f1_r_z + 1],
        ], axis=1)

        faces_2_l = np.stack([
            v_idx[f2_l_x, f2_l_y, f2_l_z],
            v_idx[f2_l_x + 1, f2_l_y, f2_l_z],
            v_idx[f2_l_x + 1, f2_l_y, f2_l_z + 1],
            v_idx[f2_l_x, f2_l_y, f2_l_z + 1],
        ], axis=1)

        faces_2_r = np.stack([
            v_idx[f2_r_x, f2_r_y, f2_r_z],
            v_idx[f2_r_x, f2_r_y, f2_r_z + 1],
            v_idx[f2_r_x + 1, f2_r_y, f2_r_z + 1],
            v_idx[f2_r_x + 1, f2_r_y, f2_r_z],
        ], axis=1)

        faces_3_l = np.stack([
            v_idx[f3_l_x, f3_l_y, f3_l_z],
            v_idx[f3_l_x, f3_l_y + 1, f3_l_z],
            v_idx[f3_l_x + 1, f3_l_y + 1, f3_l_z],
            v_idx[f3_l_x + 1, f3_l_y, f3_l_z],
        ], axis=1)

        faces_3_r = np.stack([
            v_idx[f3_r_x, f3_r_y, f3_r_z],
            v_idx[f3_r_x + 1, f3_r_y, f3_r_z],
            v_idx[f3_r_x + 1, f3_r_y + 1, f3_r_z],
            v_idx[f3_r_x, f3_r_y + 1, f3_r_z],
        ], axis=1)

        faces = np.concatenate([
            faces_1_l, faces_1_r,
            faces_2_l, faces_2_r,
            faces_3_l, faces_3_r,
        ], axis=0)

        vertices = self.loc + self.scale * vertices
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        return mesh

    @property
    def resolution(self):
        assert(self.data.shape[0] == self.data.shape[1] == self.data.shape[2])
        return self.data.shape[0]

    def contains(self, points):
        nx = self.resolution

        # Rescale bounding box to [-0.5, 0.5]^3
        points = (points - self.loc) / self.scale
        # Discretize points to [0, nx-1]^3
        points_i = ((points + 0.5) * nx).astype(np.int32)
        # i1, i2, i3 have sizes (batch_size, T)
        i1, i2, i3 = points_i[..., 0],  points_i[..., 1],  points_i[..., 2]
        # Only use indices inside bounding box
        mask = (
            (i1 >= 0) & (i2 >= 0) & (i3 >= 0)
            & (nx > i1) & (nx > i2) & (nx > i3)
        )
        # Prevent out of bounds error
        i1 = i1[mask]
        i2 = i2[mask]
        i3 = i3[mask]

        # Compute values, default value outside box is 0
        occ = np.zeros(points.shape[:-1], dtype=np.bool)
        occ[mask] = self.data[i1, i2, i3]

        return occ