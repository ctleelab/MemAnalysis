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

import numpy as np
import MDAnalysis as mda

import MemAnalysis
from MemAnalysis.base_spectral_analysis import MembraneSpectralAnalysis
from MemAnalysis.surface import get_z_surface, get_interpolated_z_surface
from MemAnalysis.leaflet_finder import determine_leaflets

# Load real data
topology_file = "tests/sys.gro"
trajectory_file = "tests/prod_center_skip1000.xtc"
u = mda.Universe(topology_file, trajectory_file)

MemAnalysis.apl.area_per_lipid(u)


# Surface Tests
def test_get_z_surface_shape():
    coords = u.atoms.positions
    z_surface = get_z_surface(
        coords,
        n_x_bins=10,
        n_y_bins=10,
        x_range=(0, u.dimensions[0]),
        y_range=(0, u.dimensions[1]),
    )
    assert z_surface.shape == (10, 10)


def test_interpolated_surface():
    coords = u.atoms.positions
    P, Q = np.mgrid[0 : u.dimensions[0] : 2, 0 : u.dimensions[1] : 2]
    ag = u.select_atoms("all")
    z_interp = get_interpolated_z_surface(coords, P, Q, ag=ag)
    assert z_interp.shape == P.shape


# Leaflet Tests
def test_determine_leaflets():
    leaflets = determine_leaflets(u, selection="name P*")
    assert "upper" in leaflets and "lower" in leaflets
    selected_atoms = u.select_atoms("name P*")
    assert len(leaflets["upper"]) + len(leaflets["lower"]) == len(selected_atoms)


# Not the place to do it but create a test for flip detection in leaflet related test
# # Spectral Analysis Tests
# # I have to define better the inputs in MembraneSpectralAnalysis
# def test_membrane_spectral_analysis():
#     msa = MembraneSpectralAnalysis(u, select='name P*', n_x_bins=10, n_y_bins=10)
#     msa.run(start=0, stop=1)
#     assert not np.isnan(msa.results.thickness[0]).all()
#     assert not np.isnan(msa.results.height_power_spectrum[0]).all()
