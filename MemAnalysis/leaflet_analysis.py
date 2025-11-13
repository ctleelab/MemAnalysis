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
        phosphate_atoms.append(f"(resname {resname} and name P*)")  # selects atoms with 'P' in name
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


def leaflet_residue_counts_per_frame(u: mda.Universe) -> List[Tuple[float, int, int, dict, dict]]:
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
            raise ValueError(f"LeafletFinder found fewer than two groups (n={n_groups}).")

        # Compute leaflets for this frame
        leaflets = LeafletFinder(u, selection_string, rcutoff, pbc=True)
        top = leaflets.groups(0)
        bottom = leaflets.groups(1)

        # Count residues
        top_count = len(top.residues)
        bottom_count = len(bottom.residues)
        top_residue_counts = util.count_residues(top.residues)
        bottom_residue_counts = util.count_residues(bottom.residues)
        
        results.append((ts.time, top_count, bottom_count, top_residue_counts, bottom_residue_counts))

    return results
