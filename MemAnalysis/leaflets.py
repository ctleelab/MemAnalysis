#
# MemAnalysis
#
# Copyright 2025- The MemAnalysis Authors
# and the project initiators Carolina Sarto and Christopher T. Lee.
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Please help us support development by citing the research
# papers on the package. Check out https://github.com/ctleelab/MemAnalysis/
# for more information.

import numpy as np
from MDAnalysis.analysis.leaflet import LeafletFinder, optimize_cutoff
import MDAnalysis as mda

def get_leaflets(atoms, cutoff=None, verbose=True):
    """Split `atoms` (an AtomGroup) into upper/lower leaflets."""
    if cutoff is None:
        cutoff, _ = optimize_cutoff(atoms.universe, atoms)

    finder = LeafletFinder(atoms.universe, atoms, cutoff=cutoff, pbc=True)

    # Two largest connected components should be the two leaflets
    sizes = finder.sizes()
    a, b = sorted(sizes, key=sizes.get, reverse=True)[:2]
    g1, g2 = finder.group(a), finder.group(b)
    upper, lower = (g1, g2) if g1.centroid()[2] > g2.centroid()[2] else (g2, g1)

    if verbose:
        n_missing = len(atoms) - (len(upper) + len(lower))
        if n_missing:
            print(
                f"Warning: {n_missing} atoms not assigned to either leaflet "
                f"({len(sizes)} groups found, only the 2 largest were kept)"
            )
        if len(upper) != len(lower):
            print(
                f"Warning: leaflets have different sizes "
                f"(upper={len(upper)}, lower={len(lower)})"
            )

    return upper, lower, cutoff


def get_leaflets_trajectory(
    topology,
    trajectory,
    select="name PO4* or name P",     # name PO4* applies to marini beads and name P applies to atomistic simulations.
    equil_frames=0,
    stride=1,
    imbalance_thresh=0.15,
    gap_thresh=2.0,
):
    """Cheap mean-z leaflet split per frame, tracking flips against the first
    successfully-processed frame. Stops early and returns what it has so far
    if a frame looks unreliable (size imbalance, no gap at the midplane, or z
    wrapped across the periodic boundary) — use `leaflets_via_anchors`
    instead if that happens. Otherwise returns the upper/lower resids from
    the last frame processed.
    """
    u = mda.Universe(topology, trajectory)
    heads = u.select_atoms(select)
    frames = list(range(equil_frames, len(u.trajectory), stride))

    ref_frame, ref_upper, ref_lower = None, None, None
    upper_resids, lower_resids = None, None

    for i in frames:
        u.trajectory[i]
        z = heads.positions[:, 2]
        zmean = z.mean()
        top_mask = z > zmean

        imbalance = abs(2 * top_mask.sum() - len(z)) / len(z)

        sorted_z = np.sort(z)
        mid = np.searchsorted(sorted_z, zmean)
        gap = sorted_z[mid] - sorted_z[mid - 1] if 0 < mid < len(sorted_z) else 0.0

        box_z = u.trajectory.ts.dimensions[2]
        wrapped = (z.max() - z.min()) > 0.9 * box_z

        if imbalance > imbalance_thresh or gap < gap_thresh or wrapped:
            print(f"Warning: mean-z split unreliable at frame {i}, stopping")
            break

        upper_resids = set(heads.resindices[top_mask])
        lower_resids = set(heads.resindices[~top_mask])

        if ref_upper is None:
            ref_frame, ref_upper, ref_lower = i, upper_resids, lower_resids
        elif upper_resids != ref_upper or lower_resids != ref_lower:
            print(f"I found a flip in frame {i} (leaflets differ from frame {ref_frame})")

    return upper_resids, lower_resids


def leaflets_via_anchors(topology, trajectory, select="name PO4* or name P"):
    """LeafletFinder-based leaflet split at the first, middle and last frame.
    Checks leaflet-size homogeneity (via `get_leaflets`) and whether leaflet
    identity flips between anchors, then returns the upper/lower resids
    (from the first frame).
    """
    u = mda.Universe(topology, trajectory)
    heads = u.select_atoms(select)

    n_frames = len(u.trajectory)
    anchors = sorted({0, n_frames // 2, n_frames - 1})

    upper_resids, lower_resids, cutoff = {}, {}, None
    for i in anchors:
        u.trajectory[i]
        upper, lower, cutoff = get_leaflets(heads, cutoff=cutoff)
        upper_resids[i] = {r.resindex for r in upper.residues}
        lower_resids[i] = {r.resindex for r in lower.residues}

    ref = anchors[0]
    for i in anchors[1:]:
        if upper_resids[i] != upper_resids[ref] or lower_resids[i] != lower_resids[ref]:
            print(f"Warning: leaflets flipped between frame {ref} and frame {i}")

    return upper_resids[ref], lower_resids[ref]
