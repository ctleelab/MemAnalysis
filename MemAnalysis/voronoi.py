"""Voronoi tessellation: 
1- True polygon areas.
1- nearest-generator rasterization.
3- Cached per (row, system, variant) so multiple scripts reuse the same cKDTree assignment instead of recomputing it each."""

import pickle

import numpy as np
from scipy.spatial import cKDTree

# def voronoi_areas(xy, Lx, Ly):

def voronoi_grid(Lx, Ly, nbins):
    """Pixel-center query points for an nbins x nbins raster, row-major [y, x] to match imshow(origin='lower')."""
    xedges = np.linspace(0.0, Lx, nbins + 1)
    yedges = np.linspace(0.0, Ly, nbins + 1)
    xc = (xedges[:-1] + xedges[1:]) / 2
    yc = (yedges[:-1] + yedges[1:]) / 2
    xx, yy = np.meshgrid(xc, yc)
    return np.column_stack([xx.ravel(), yy.ravel()])


def nearest_generator_indices(xy, Lx, Ly, query_points):
    """xy are the seeds/generators. Index into `xy` of the nearest generator for each query point, periodic under boxsize=[Lx,Ly].
    float64 promotion + clamp: float32 positions can round to exactly Lx/Ly after `%`, which cKDTree's
    periodic mode rejects."""
    wrapped = xy.astype(np.float64)
    wrapped[:, 0] = np.clip(wrapped[:, 0] % Lx, 0.0, np.nextafter(Lx, 0))
    wrapped[:, 1] = np.clip(wrapped[:, 1] % Ly, 0.0, np.nextafter(Ly, 0))
    tree = cKDTree(wrapped, boxsize=[Lx, Ly])
    _, idx = tree.query(query_points)       # "_" means that we won't need the distances (first return object)
    return idx                      # idx[k] is the index of the nearest seed/generator to query_points[k]


def voronoi_raster_2d(xy, values, Lx, Ly, query_points, nbins):
    """Nearest-generator (true Voronoi cell) rasterization of `values` at lateral positions `xy`."""
    idx = nearest_generator_indices(xy, Lx, Ly, query_points)
    return values[idx].reshape(nbins, nbins)


def compute_tessellation(u, heads, protein, equil_frames, stride, nbins):
    """Per-frame nearest-generator index for both leaflets -- indices are into `heads` itself (stable
    across frames and across scripts), not into the frame's own top/bot subset, so any script computing
    its own per-generator values for the same `heads` ordering can rasterize by simple indexing."""
    u.trajectory[equil_frames]
    box0 = u.dimensions[:2]
    Lx, Ly = float(box0[0]), float(box0[1])
    edges = (0.0, Lx, 0.0, Ly)
    query_points = voronoi_grid(Lx, Ly, nbins)

    top_idx_frames, bot_idx_frames, protein_xy_list = [], [], []
    for _ in u.trajectory[equil_frames::stride]:
        pos = heads.positions
        z = pos[:, 2]
        top_mask = z > z.mean()
        top_where = np.where(top_mask)[0]
        bot_where = np.where(~top_mask)[0]
        top_local = nearest_generator_indices(pos[top_mask, :2], Lx, Ly, query_points)
        bot_local = nearest_generator_indices(pos[~top_mask, :2], Lx, Ly, query_points)
        top_idx_frames.append(top_where[top_local].reshape(nbins, nbins))
        bot_idx_frames.append(bot_where[bot_local].reshape(nbins, nbins))
        protein_xy_list.append(protein.positions[:, :2].mean(axis=0))

    return {
        "top_idx": np.array(top_idx_frames, dtype=np.int32),
        "bot_idx": np.array(bot_idx_frames, dtype=np.int32),
        "edges": edges,
        "protein_xy": np.mean(protein_xy_list, axis=0),
    }


def get_tessellation(cache_path, u, heads, protein, equil_frames, stride, nbins):
    """Load the cached tessellation at `cache_path`, or compute and save it if not present."""
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    tess = compute_tessellation(u, heads, protein, equil_frames, stride, nbins)
    with open(cache_path, "wb") as f:
        pickle.dump(tess, f, protocol=pickle.HIGHEST_PROTOCOL)
    return tess
