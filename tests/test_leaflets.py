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

import MDAnalysis as mda
import MemAnalysis as ma

topology_file = "tests/sys.gro"
trajectory_file = "tests/prod_center_skip1000.xtc"
u = mda.Universe(topology_file, trajectory_file)
phosphates = u.select_atoms("name P*")


def test_get_leaflets():
    upper, lower, cutoff = ma.lf.get_leaflets(phosphates, verbose=False)

    assert set(upper.indices).isdisjoint(lower.indices)
    assert len(upper) > 0
    assert len(lower) > 0
    assert cutoff > 0


def test_get_leaflets_trajectory():
    upper_resids, lower_resids = ma.lf.get_leaflets_trajectory(
        topology_file, trajectory_file, select="name P*"
    )

    assert isinstance(upper_resids, set)
    assert isinstance(lower_resids, set)
    assert upper_resids.isdisjoint(lower_resids)
    assert len(upper_resids) > 0
    assert len(lower_resids) > 0


def test_leaflets_via_anchors():
    upper_resids, lower_resids = ma.lf.leaflets_via_anchors(
        topology_file, trajectory_file, select="name P*"
    )

    assert isinstance(upper_resids, set)
    assert isinstance(lower_resids, set)
    assert upper_resids.isdisjoint(lower_resids)
    assert len(upper_resids) > 0
    assert len(lower_resids) > 0
