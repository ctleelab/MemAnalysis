import MDAnalysis as mda
from typing import Tuple, List
from MemAnalysis import util
import numpy as np

def area_per_lipid(u: mda.Universe) -> float:
    """
    Estimate area per lipid from the simulation box dimensions.

    Args:
        u (MDAnalysis.Universe): MDAnalysis Universe object

    Returns:
        float: estimated area per lipid in nm²
    """
    # Get box dimensions (x and y) in Å
    box = u.dimensions[:2] 
    area = box[0] * box[1]

    # Count number of lipids
    lipid_resnames = util.find_lipid_resnames(u)
    lipids = [res for res in u.residues if res.resname in lipid_resnames]
    n_lipids = len(lipids) // 2  # Divide by 2 for two leaflets

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
    lipid_resnames = util.find_lipid_resnames(u)
    lipid_residues = [res for res in u.residues if res.resname in lipid_resnames]
    n_lipids = len(lipid_residues) // 2  # Divide by 2 for two leaflets

    if n_lipids == 0:
        raise ValueError("No lipid residues found in the system.")

    area_per_lipid_values = []
    for ts in u.trajectory:
        box = ts.dimensions[:2]
        area = box[0] * box[1]             
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
    print(f"Area per lipid: {mean_apl:.3f} ± {std_apl:.3f} nm²")    
    return mean_apl, std_apl    




