import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder, optimize_cutoff

import numpy as np
import numpy.typing as npt

from typing import Tuple, Callable, List
import warnings
from scipy import stats

def system_report(u: mda.Universe) -> dict:
    """Generate a report of the system composition.

    Args:
        u (mda.Universe): MDAnalysis Universe object

    Returns:
        dict: count per resname
    """
    count_dict = {}
    for residue in u.residues:
        if residue.resname not in count_dict:
            count_dict[residue.resname] = 1
        else:
            count_dict[residue.resname] += 1
    print(f"\tComposition: {count_dict}")
    # # total_charge is not recognized for real data but works for test data
    # assert u.atoms.total_charge() == 0
    # print(f"\t     charge: {u.atoms.total_charge()}")
    print(f"\t  num atoms: {len(u.atoms)}")
    print(f"\t dimensions: {u.dimensions[0:3]/10} nm")
    print()
    return count_dict


def find_lipid_resnames(u: mda.Universe) -> set:
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

    # Count number of lipids (assuming each lipid is a residue named 'POPC', 'DPPC', etc.)
    lipid_resnames = find_lipid_resnames(u)
    lipids = [res for res in u.residues if res.resname in lipid_resnames]
    n_lipids = len(lipids)

    if n_lipids == 0:
        raise ValueError("No lipid residues found in the system.")

    return area / n_lipids

def area_per_lipid_per_frame(topology_file: str, trajectory_file: str) -> List[float]:
    """
    Calculate area per lipid for each frame in the trajectory.

    Args:
        topology_file (str): Path to topology file (e.g., sys.gro)
        trajectory_file (str): Path to trajectory file (e.g., sys.xtc)

    Returns:
        List[float]: Area per lipid for each frame in nm²
    """
    u = mda.Universe(topology_file, trajectory_file)
    lipid_resnames = find_lipid_resnames(u)
    lipid_residues = [res for res in u.residues if res.resname in lipid_resnames]
    n_lipids = len(lipid_residues)

    if n_lipids == 0:
        raise ValueError("No lipid residues found in the system.")

    area_per_lipid_values = []
    for ts in u.trajectory:
        box = ts.dimensions[:2]/10  # [lx, ly] in nm
        area = box[0] * box[1]   # xy-plane area
        area_per_lipid = area / n_lipids
        area_per_lipid_values.append(area_per_lipid)

    return area_per_lipid_values

def count_residues(u):
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


def _check_leaflet(u):
    rcutoff, n = optimize_cutoff(u, "name PO4 PO41 PO42")
    print(rcutoff, n)
    leafs = LeafletFinder(u, "name PO4 PO41 PO42", rcutoff, pbc=True)
    top = leafs.groups(0)
    bottom = leafs.groups(1)

    print(len(top.residues), count_residues(top))
    print(len(bottom.residues), count_residues(bottom))

    return (set([r.ix for r in top.residues]), set([r.ix for r in bottom.residues]))


def check_leaflet(top, gro):
    u = mda.Universe(top, gro, topology_format="ITP")
    return _check_leaflet(u)

def leaflet_residue_counts_per_frame(topology_file: str, trajectory_file: str) -> List[float]:
    """
    Perform leaflet analysis per frame and return the number of residues
    in top and bottom leaflets for each frame.

    Args:
        topology_file (str): Path to topology file (e.g., sys.gro)
        trajectory_file (str): Path to trajectory file (e.g., sys.xtc)

    Returns:
        list of tuples: [(time, top_count, bottom_count), ...]
    """
    u = mda.Universe(topology_file, trajectory_file)
    results = []

    for ts in u.trajectory:
        rcutoff, _ = optimize_cutoff(u, "name PO4 PO41 PO42")
        leaflets = LeafletFinder(u, "name PO4 PO41 PO42", rcutoff, pbc=True)
        top = leaflets.groups(0)
        bottom = leaflets.groups(1)

        results.append((ts.time, len(top.residues), len(bottom.residues),count_residues(top), count_residues(bottom)))

    return results
