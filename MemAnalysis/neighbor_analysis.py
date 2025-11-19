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

"""
This script performs lipid-lipid neighbor analysis in membrane simulations using:

Distance-based method:
   - Defines neighbors within a specified cutoff (e.g., 15 Å) using MDAnalysis selection language.
   - Captures enrichment of lipids in proximity across frames.

This produce the following output structures:
    {
        "upper": ndarray (frames x lipids x lipids),
        "lower": ndarray (frames x lipids x lipids),
        "name_map": dict mapping lipid names to indices
    }
"""

import numpy as np
from scipy.spatial import Voronoi
from tqdm.auto import tqdm
import MDAnalysis as mda
from pathlib import Path
from MemAnalysis.util import find_lipid_resnames
from MemAnalysis.leaflet_analysis import leaflet_residue_counts


def run_neighbor_search(u: mda.Universe, cutoff: float = 15.0) -> dict:
    """
    Compute distance-based lipid neighbor enrichment per frame for both leaflets

    Args:
    u (MDAnalysis.Universe): Universe object.
    cutoff (float, optional): Distance cutoff in Å for neighbor search. Defaults to 15.0.

    Returns:
        dict: {
            "upper": ndarray (frames x lipids x lipids),
            "lower": ndarray (frames x lipids x lipids),
            "name_map": dict mapping lipid names to indices
        }
    """
    # Detect lipids dynamically
    lipid_resnames = find_lipid_resnames(u)
    lipid_dict = {name: i for i, name in enumerate(lipid_resnames)}

    # Identify leaflets dynamically
    top_resids, bottom_resids = leaflet_residue_counts(u)
    upper_atoms = u.residues[list(top_resids)].atoms
    lower_atoms = u.residues[list(bottom_resids)].atoms
    leaflets = {"upper": upper_atoms, "lower": lower_atoms}

    # Initialize count arrays
    count_dict = {
        "upper": np.zeros(
            (len(u.trajectory), len(lipid_resnames), len(lipid_resnames))
        ),
        "lower": np.zeros(
            (len(u.trajectory), len(lipid_resnames), len(lipid_resnames))
        ),
        "name_map": lipid_dict,
    }

    # Loop through frames
    for ts in tqdm(u.trajectory, desc="Neighbor search"):
        for leaflet in ["upper", "lower"]:
            for atom in leaflets[leaflet].atoms:
                i = lipid_dict[atom.resname]
                curr = mda.AtomGroup([atom])
                sel = leaflets[leaflet].select_atoms(
                    f"around {cutoff} group curr", curr=curr, updating=True
                )
                for res in sel.residues:
                    j = lipid_dict[res.resname]
                    if i <= j:
                        count_dict[leaflet][ts.frame, i, j] += 1
                    else:
                        count_dict[leaflet][ts.frame, j, i] += 1

    return count_dict
