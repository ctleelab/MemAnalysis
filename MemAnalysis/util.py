import MDAnalysis as mda

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
    print(f"Time range: {time_range_ps[0]/1000} ns to {time_range_ps[1]/1000} ns")
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


