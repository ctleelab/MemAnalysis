import pytest
import MemAnalysis as ma
import MDAnalysis as mda
from pathlib import Path

# Test files
topology_file = "tests/sys.gro"
trajectory_file = "tests/prod_center_skip1000.xtc"
u = mda.Universe(topology_file, trajectory_file)

def test_run_voronoi():
    result = ma.na.run_voronoi(u)
    
    # Check that result is a dictionary
    assert isinstance(result, dict)
    
    # Check keys
    assert "upper" in result and "lower" in result and "name_map" in result
    
    # Check array shapes
    n_frames = len(u.trajectory)
    n_lipids = len(result["name_map"])
    assert result["upper"].shape == (n_frames, n_lipids, n_lipids)
    assert result["lower"].shape == (n_frames, n_lipids, n_lipids)
    
    # Check that pickle file exists in .tmp
    tmp_dir = Path(u.filename).parent / ".tmp"
    output_file = tmp_dir / f"{Path(u.filename).stem}_voronoi_leaflet_glo.pickle"
    assert output_file.exists()

def test_run_neighbor_search_default_cutoff():
    result = ma.na.run_neighbor_search(u)  # uses default cutoff = 15.0
    
    # Check that result is a dictionary
    assert isinstance(result, dict)
    
    # Check keys
    assert "upper" in result and "lower" in result and "name_map" in result
    
    # Check array shapes
    n_frames = len(u.trajectory)
    n_lipids = len(result["name_map"])
    assert result["upper"].shape == (n_frames, n_lipids, n_lipids)
    assert result["lower"].shape == (n_frames, n_lipids, n_lipids)
    
    # Check that pickle file exists in .tmp
    tmp_dir = Path(u.filename).parent / ".tmp"
    output_file = tmp_dir / f"{Path(u.filename).stem}_neighbor_enrichment_leaflet_glo.pickle"
    assert output_file.exists()

def test_run_neighbor_search_custom_cutoff():
    result = ma.na.run_neighbor_search(u, cutoff=10.0)
    
    # Check that result is a dictionary
    assert isinstance(result, dict)
    
    # Check keys
    assert "upper" in result and "lower" in result and "name_map" in result
    
    # Check array shapes
    n_frames = len(u.trajectory)
    n_lipids = len(result["name_map"])
    assert result["upper"].shape == (n_frames, n_lipids, n_lipids)
    assert result["lower"].shape == (n_frames, n_lipids, n_lipids)