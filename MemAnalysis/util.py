import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder, optimize_cutoff

import numpy as np
import numpy.typing as npt

from typing import Tuple, Callable, List
import warnings
from scipy import stats

def count_residues(u):
    """
    Count residues in the given Universe, distinguishing ions by atom name.

    Args:
         u (mda.Universe): MDAnalysis Universe object 
    
    Returns:
        dict: Dictionary with residue names as keys and their counts as values.

    """
    count_dict = {}
    for residue in u.residues:
        if residue.resname == "ION":
            name = residue.atoms[0].name
            if name not in count_dict:
                count_dict[name] = 1
            else:
                count_dict[name] += 1
        else:
            if residue.resname not in count_dict:
                count_dict[residue.resname] = 1
            else:
                count_dict[residue.resname] += 1
    return count_dict

def system_report(u: mda.Universe) -> dict:
    """
    Print a formatted system report to the console.
    """
    composition = count_residues(u.residues)
    num_atoms = len(u.atoms)
    num_residues = len(u.residues)
    box_dimensions_nm = (u.dimensions[0:3] / 10).tolist()  # convert Å to nm
    n_frames = len(u.trajectory)
    time_range_ps = (u.trajectory[0].time, u.trajectory[-1].time)

    print("=== SYSTEM REPORT ===")
    print(f"Composition: {composition}")
    print(f"Number of atoms: {num_atoms}")
    print(f"Number of residues: {num_residues}")
    print(f"Box dimensions: {box_dimensions_nm} nm")
    print(f"Frames: {n_frames}")
    print(f"Time range: {time_range_ps[0]} ps to {time_range_ps[1]} ps")
    print("======================")



def find_lipid_resnames(u: mda.Universe) -> set:
    """
    Identify lipid residue names in the system by looking for residues
    that contain phosphate atoms (atom name containing "P"). 
    
    There might be not the best way to do it, but it works for common lipids
    when you don't have other P-containing molecules in the system.

    Args:
        u (MDAnalysis.Universe): MDAnalysis Universe object 
    
    Returns:
        set: set of lipid residue names

    """
    lipid_resnames = set()
    for residue in u.residues:
        for atom in residue.atoms:
            if "P" in atom.name:
                lipid_resnames.add(residue.resname)
                break  # No need to check other atoms in this residue
    print(f"Identified lipid resnames: {lipid_resnames}")
    return lipid_resnames


def area_per_lipid(u: mda.Universe) -> float:
    """
    Estimate area per lipid from the simulation box dimensions.

    Args:
        u (MDAnalysis.Universe): MDAnalysis Universe object

    Returns:
        float: estimated area per lipid in nm²
    """
    # Get box dimensions (x and y) in Å
    box = u.dimensions[:2]/10  # [lx, ly] nm
    area = box[0] * box[1]

    # Count number of lipids
    lipid_resnames = find_lipid_resnames(u)
    lipids = [res for res in u.residues if res.resname in lipid_resnames]
    n_lipids = len(lipids)

    if n_lipids == 0:
        raise ValueError("No lipid residues found in the system.")

    return area / n_lipids

def area_per_lipid_per_frame(u: mda.Universe) -> List[float]:
    """
    Calculate area per lipid for each frame in the trajectory.

    Args:
        u (MDAnalysis.Universe): Universe object containing topology and trajectory.

    Returns:
        List[float]: Area per lipid for each frame in nm²
    """
    lipid_resnames = find_lipid_resnames(u)
    lipid_residues = [res for res in u.residues if res.resname in lipid_resnames]
    n_lipids = len(lipid_residues)

    if n_lipids == 0:
        raise ValueError("No lipid residues found in the system.")

    area_per_lipid_values = []
    for ts in u.trajectory:
        box = ts.dimensions[:2] / 10  # Convert from Å to nm
        area = box[0] * box[1]        # xy-plane area
        area_per_lipid = area / n_lipids
        area_per_lipid_values.append(area_per_lipid)
    
    return area_per_lipid_values

def area_per_lipid_stats(u: mda.Universe) -> Tuple[float, float]:
    """
    Calculate mean and standard deviation of area per lipid over the trajectory.

    Args:
        u (MDAnalysis.Universe): Universe object containing topology and trajectory.

    Returns:
        Tuple[float, float]: Mean and standard deviation of area per lipid in nm²
    """
    apl_values = area_per_lipid_per_frame(u)
    mean_apl = np.mean(apl_values)
    std_apl = np.std(apl_values)
    return mean_apl, std_apl    


def check_leaflet(u: mda.Universe) -> Tuple[Set[int], Set[int]]:
    """
    Identify top and bottom leaflets in a bilayer system using phosphate-containing lipid residues.

    Args:
        u (MDAnalysis.Universe): MDAnalysis Universe object

    Returns:
        Tuple[Set[int], Set[int]]: Sets of residue indices (r.ix) for top and bottom leaflets.
    """
    # Find lipid residue names
    lipid_resnames = find_lipid_resnames(u)

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
    lipid_resnames = find_lipid_resnames(u)

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
        top_residue_counts = count_residues(top.residues)
        bottom_residue_counts = count_residues(bottom.residues)

        results.append((ts.time, top_count, bottom_count, top_residue_counts, bottom_residue_counts))

    return results

