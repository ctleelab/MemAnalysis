### If the function stats with test_ it will be automatically run by pytest

import pytest
import MemAnalysis as ma
import MDAnalysis as mda


def test_dummy_function():
    result = ma.dummy.dummy_function(16)
    assert result == 4.0

def test_system_report():
    topology_file = "tests/sys.gro"
    trajectory_file = "tests/prod_center_skip1000.xtc"
    u = mda.Universe(topology_file, trajectory_file)
    report = ma.util.system_report(u)
    assert report['TOCL'] == 760

def test_area_per_lipid_per_frame():
    topology_file = "tests/sys.gro"
    trajectory_file = "tests/prod_center_skip1000.xtc"
    report = ma.util.area_per_lipid_per_frame(topology_file, trajectory_file)
    print(report)

def test_count_residues():
    u = mda.Universe("tests/sys.gro", "tests/prod_center_skip1000.xtc")
    counts = ma.util.count_residues(u)
    assert isinstance(counts, dict)
    assert all(isinstance(k, str) for k in counts.keys())
    assert all(isinstance(v, int) for v in counts.values())

def test_check_leaflet():
    u = mda.Universe("tests/sys.gro", "tests/prod_center_skip1000.xtc")
    top, bottom = ma.util._check_leaflet(u)
    assert isinstance(top, set)
    assert isinstance(bottom, set)
    assert top.isdisjoint(bottom)
    assert len(top) > 0
    assert len(bottom) > 0

def test_leaflet_residue_counts_per_frame():
    topology_file = "tests/sys.gro"
    trajectory_file = "tests/prod_center_skip1000.xtc"
    report = ma.util.leaflet_residue_counts_per_frame(topology_file, trajectory_file)
    print(report) 