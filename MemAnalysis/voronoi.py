#
# MemAnalysis
#
# Copyright 2026- The MemAnalysis Authors
# and the project initiators Carolina Sarto and Christopher T. Lee.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Please help us support development by citing the research
# papers on the package. Check out https://github.com/ctleelab/MemAnalysis/
# for more information.

"""Voronoi tessellation.
- periodic_voronoi: exact tessellation (scipy.spatial.Voronoi) of a set of generators, periodically tiled so edge cells are correct.
- voronoi_grid: pixel-center query points for an nbins x nbins raster.
- nearest_generator_indices: nearest generator (by index) for each query point, periodic under boxsize=[Lx,Ly].
- voronoi_raster_2d: rasterize per-generator values onto the grid via nearest-generator assignment.
- compute_tessellation / get_tessellation: per-frame nearest-generator indices for both leaflets, cached so multiple scripts reuse the same cKDTree assignment instead of recomputing it each.
"""

import pickle

import numpy as np
from scipy.spatial import Voronoi, cKDTree


def periodic_voronoi(gen_xy, Lx, Ly):
    """Exact Voronoi tessellation (scipy.spatial.Voronoi) of generators `gen_xy`, periodically
    tiled (3x3 copies) so cells at the box edge are geometrically correct. `gen_xy` should
    already be wrapped into [0, Lx) x [0, Ly)."""
    shifts = [(dx * Lx, dy * Ly) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
    tiled_xy = np.vstack([gen_xy + shift for shift in shifts])
    return Voronoi(tiled_xy)


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
    _, idx = tree.query(
        query_points
    )  # "_" means that we won't need the distances (first return object)
    return idx  # idx[k] is the index of the nearest seed/generator to query_points[k]


def voronoi_raster_2d(xy, values, Lx, Ly, query_points, nbins):
    """Nearest-generator (true Voronoi cell) rasterization of `values` at lateral positions `xy`."""
    idx = nearest_generator_indices(xy, Lx, Ly, query_points)
    return values[idx].reshape(nbins, nbins)


def compute_tessellation(u, heads, protein, equil_frames, stride, nbins):
    """Per-frame nearest-generator index for both leaflets -- indices are into `heads` itself (stable
    across frames and across scripts), not into the frame's own top/bot subset, so any script computing
    its own per-generator values for the same `heads` ordering can rasterize by simple indexing.
    """
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
