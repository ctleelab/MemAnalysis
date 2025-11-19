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
