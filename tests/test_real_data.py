### If the function stats with test_ it will be automatically run by pytest

import pytest
import MemAnalysis as ma
import MDAnalysis as mda

topology_file = "tests/sys.gro"
trajectory_file = "tests/prod_center_skip1000.xtc"
u = mda.Universe(topology_file, trajectory_file)

def test_dummy_function():
    result = ma.dummy.dummy_function(16)
    assert result == 4.0

def test_system_report():
    report = ma.util.system_report(u)
    # assert report['TOCL'] == 760
    # print(report)

def test_area_per_lipid_per_frame():
    report = ma.apl.area_per_lipid_per_frame(u)
    # print(report)

def test_area_per_lipid_stats():
    mean_area, std_area = ma.apl.area_per_lipid_stats(u)
    # print(f"Mean area per lipid: {mean_area} nm²")
    # print(f"Std area per lipid: {std_area} nm²")

def test_count_residues():
    counts = ma.util.count_residues(u)
    # Confirms that the result is a dictionary
    assert isinstance(counts, dict)

    # Confirms that keys are strings and values are integers
    assert all(isinstance(k, str) for k in counts.keys())
    assert all(isinstance(v, int) for v in counts.values())

def test_leaflet_residue_counts():
    top, bottom = ma.la.leaflet_residue_counts(u)
    
    # Verifies that the both leaflets is returned as a set.
    assert isinstance(top, set)
    assert isinstance(bottom, set)

    # Verifies that there is no overlap between the two leaflets
    assert top.isdisjoint(bottom)
    
    # Verifies that both leaflets are not empty
    assert len(top) > 0
    assert len(bottom) > 0
    print(f"Top leaflet total residues: {len(top)}")
    print(f"Bottom leaflet total residues: {len(bottom)}") 

def test_leaflet_residue_counts_per_frame():
    report = ma.la.leaflet_residue_counts_per_frame(u)
    print(report[:2]) 