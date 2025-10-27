

import pytest
import MemAnalysis as ma
import MDAnalysis as mda
from MDAnalysisData import datasets

def test_dummy_function():
    result = ma.dummy.dummy_function(16)
    assert result == 4.0


def test_system_report():
    data = datasets.fetch_membrane_peptide()
    u = mda.Universe(data.topology, data.trajectory)
    report = ma.util.system_report(u)
    assert report['ALA'] == 9
