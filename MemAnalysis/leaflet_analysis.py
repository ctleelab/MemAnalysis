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

import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder, optimize_cutoff
from typing import Set, Tuple, List
from MemAnalysis import util


def leaflet_residue_counts(u: mda.Universe) -> Tuple[Set[int], Set[int]]:
    """
    Identify top and bottom leaflets in a bilayer system using phosphate-containing lipid residues.

    Args:
        u (MDAnalysis.Universe): MDAnalysis Universe object

    Returns:
        Tuple[Set[int], Set[int]]: Sets of residue indices (r.ix) for top and bottom leaflets.
    """
    # Find lipid residue names
    lipid_resnames = util.find_lipid_resnames(u)

    # Build a selection string for phosphate atoms in those residues
    phosphate_atoms = []
    for resname in lipid_resnames:
        phosphate_atoms.append(
            f"(resname {resname} and name P*)"
        )  # selects atoms with 'P' in name
    selection_string = " or ".join(phosphate_atoms)

    # Use optimized cutoff for leaflet separation
    rcutoff, n_groups = optimize_cutoff(u, selection_string)
    if n_groups < 2:
        raise ValueError(f"LeafletFinder found fewer than two groups (n={n_groups}).")

    leafs = LeafletFinder(u, selection_string, rcutoff, pbc=True)
    top = leafs.groups(0)
    bottom = leafs.groups(1)

    top_resids = {res.ix for res in top.residues}
    bottom_resids = {res.ix for res in bottom.residues}

    return top_resids, bottom_resids


def leaflet_residue_counts_per_frame(
    u: mda.Universe,
) -> List[Tuple[float, int, int, dict, dict]]:
    """
    Perform leaflet analysis per frame and return the number of residues
    in top and bottom leaflets for each frame. Recomputes leaflets every frame
    to detect lipid flipping.

    Args:
        u (MDAnalysis.Universe): Universe object containing topology and trajectory.

    Returns:
        List[Tuple]: Each tuple contains (time, top_count, bottom_count, top_residue_counts, bottom_residue_counts)
    """
    # Dynamically find lipid residue names
    lipid_resnames = util.find_lipid_resnames(u)

    # Build selection string for phosphate atoms
    phosphate_atoms = [f"(resname {resname} and name P*)" for resname in lipid_resnames]
    selection_string = " or ".join(phosphate_atoms)

    results = []

    for ts in u.trajectory:
        # Optimize cutoff for this frame (or reuse a fixed cutoff for speed)
        rcutoff, n_groups = optimize_cutoff(u, selection_string)
        if n_groups < 2:
            raise ValueError(
                f"LeafletFinder found fewer than two groups (n={n_groups})."
            )

        # Compute leaflets for this frame
        leaflets = LeafletFinder(u, selection_string, rcutoff, pbc=True)
        top = leaflets.groups(0)
        bottom = leaflets.groups(1)

        # Count residues
        top_count = len(top.residues)
        bottom_count = len(bottom.residues)
        top_residue_counts = util.count_residues(top.residues)
        bottom_residue_counts = util.count_residues(bottom.residues)

        results.append(
            (
                ts.time,
                top_count,
                bottom_count,
                top_residue_counts,
                bottom_residue_counts,
            )
        )

    return results


def check_lipid_flip(u: mda.Universe, selection: str) -> bool:
    """
    Check for lipid flip-flop events in the trajectory by analyzing leaflet assignments per frame.

    Args:
        u (MDAnalysis.Universe): Universe object containing topology and trajectory.
        selection (str): Atom selection string to identify lipid headgroup atoms.

    Returns:
        bool: True if any lipid flip-flop events are detected, False otherwise.
    """
    # Store initial leaflet assignments
    initial_top, initial_bottom = leaflet_residue_counts(u)

    for ts in u.trajectory:
        current_top, current_bottom = leaflet_residue_counts(u)

        # Check for any residue that has changed leaflets
        flipped = initial_top.intersection(
            current_bottom
        ) or initial_bottom.intersection(current_top)

        if flipped:
            return True  # Flip-flop detected
            # Improve this to save details in a written file or log
            print(f"Flip-flop detected at time {ts.time} ps for residues: {flipped}")

    return False  # No flip-flop detected
