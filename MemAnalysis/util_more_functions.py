import MDAnalysis as mda
from MDAnalysis.analysis.leaflet import LeafletFinder, optimize_cutoff

import numpy as np
import numpy.typing as npt

from typing import Tuple, Callable, List
import warnings
from scipy import stats

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
    # ag = u.select_atoms("resname POPC DOPC POPE DOPE")
    # u.trajectory.add_transformations(center_membrane(ag, shift=5))
    # print('Centered')
    rcutoff, n = optimize_cutoff(u, "name PO4")
    print(rcutoff, n)
    leafs = LeafletFinder(u, "name PO4", rcutoff, pbc=True)
    top = leafs.groups(0)
    bottom = leafs.groups(1)
    # leafs.write_selection('selection.vmd')

    print(len(top.residues), count_residues(top))
    print(len(bottom.residues), count_residues(bottom))

    return (set([r.ix for r in top.residues]), set([r.ix for r in bottom.residues]))


def check_leaflet(top, gro):
    u = mda.Universe(top, gro, topology_format="ITP")
    return _check_leaflet(u)


def statistical_inefficiency(
    data,
    blocks: npt.NDArray[np.int32] = np.arange(1, 257, 1),
    discards: npt.NDArray[np.int32] = np.arange(0, 100, 10),
):
    SI = np.zeros((len(discards), len(blocks)))

    for i, discard in enumerate(discards):
        # Discard front bit of data
        _, remainder = np.split(data, [int(discard / 100 * len(data))])
        _, block_var, _ = _block_average(remainder, blocks)

        SI[i] = blocks * (block_var / block_var[0])
    return discards, blocks, SI


def _block_average(
    data: npt.ArrayLike, blocks: npt.NDArray[np.int32] = np.arange(1, 100, 1)
) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]:
    block_mean = np.zeros((len(blocks)))
    block_var = np.zeros((len(blocks)))
    block_sem = np.zeros((len(blocks)))
    for i, block in enumerate(blocks):
        split_indices = np.arange(block, len(data), block, dtype=int)
        if block > len(data):
            block_mean[i] = np.nan
            block_var[i] = np.nan
            block_sem[i] = np.nan
            continue
        blocked_data = np.fromiter(
            map(
                partial(np.mean, axis=0),
                np.split(data, split_indices),
            ),
            dtype=np.double,
        )

        # Truncate number of blocks if not evenly divisible
        if len(data) % block:
            blocked_data = blocked_data[:-1]
        block_mean[i] = np.mean(blocked_data)
        block_var[i] = np.var(blocked_data, ddof=1)
        block_sem[i] = stats.sem(blocked_data, ddof=1)
    return block_mean, block_var, block_sem


def block_average(
    data,
    discard=20,
    blocks: npt.NDArray[np.int32] = np.arange(1, 257, 1),
) -> Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]:
    _, remainder = np.split(data, [int(discard / 100 * len(data))])
    return _block_average(remainder, blocks)


def nd_block_average(
    data: npt.ArrayLike,
    axis: int = 0,
    func: Callable[[npt.ArrayLike], float] = np.mean,
    blocks: npt.NDArray[np.int32] = np.arange(1, 100, 1),
) -> npt.ArrayLike:
    """Perform block analysis on n-dimensional data

    Args:
        data (npt.ArrayLike): _description_
        axis (int, optional): _description_. Defaults to 0.
        func (Callable[[npt.ArrayLike], float], optional): _description_. Defaults to np.mean.
        blocks (npt.NDArray[np.int32], optional): _description_. Defaults to np.arange(1, 100, 1).

    Raises:
        np.AxisError: _description_

    Returns:
        Tuple[npt.NDArray[np.double], npt.NDArray[np.double], npt.NDArray[np.double]]: _description_
    """

    # Guard against bad axis
    if axis >= len(data.shape):
        raise np.AxisError(axis, len(data.shape))

    result_shape = tuple([v for i, v in enumerate(data.shape) if i != axis])

    result = np.empty((len(blocks), *result_shape))
    # print("results_shape", result.shape, result_shape)

    for i, block in enumerate(blocks):
        Nb, r = divmod(data.shape[axis], block)
        split_indices = np.arange(r, data.shape[axis], block, dtype=int)

        if block > data.shape[axis]:
            result[i] = np.nan
            continue

        # Compute block average
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            blocked_data = np.fromiter(
                map(
                    partial(np.mean, axis=axis),
                    np.split(data, split_indices, axis=axis),  # List of blocks
                ),
                dtype=np.dtype((np.double, (*result_shape,))),
            )

        # Truncate first block which is either empty or has less than block elements
        result[i] = func(blocked_data[1:], axis=0)
    return result.T


def parametric_bootstrap(
    rvs: List[Callable],
    n_samples: int = 9999,
) -> npt.ArrayLike:
    """Resample data given a set of distributions

    Args:
        rvs (List[Callable]): list of random valuable generators
        n_samples (int, optional): Number of samples to generate of the set. Defaults to 9999.

    Returns:
        npt.ArrayLike: Array of results
    """
    res = np.empty((len(rvs), n_samples))

    for i, rv in enumerate(rvs):
        res[i] = rv(size=n_samples)

    return res


def mean_curvature(Z, h):
    """
    Calculates mean curvature from Z cloud points.


    Parameters
    ----------
    Z: np.ndarray.
        Multidimensional array of shape (n,n).
    h: float.
        Regular grid separation

    Returns
    -------
    H : np.ndarray.
        The result of mean curvature of Z. Returns multidimensional
        array object with values of mean curvature of shape `(n, n)`.

    """

    Zx, Zy = np.gradient(Z, h)
    Zxx, Zxy = np.gradient(Zx, h)
    _, Zyy = np.gradient(Zy, h)

    H = (1 + Zx**2) * Zyy + (1 + Zy**2) * Zxx - 2 * Zx * Zy * Zxy
    H = -H / (2 * (1 + Zx**2 + Zy**2) ** (1.5))

    return H


def gaussian_curvature(Z, h):
    """
    Calculate Gaussian curvature from Z cloud points.


    Parameters
    ----------
    Z: np.ndarray.
        Multidimensional array of shape (n,n).
    varargs : list of scalar or array, optional
        Spacing between f values. Default unitary spacing for all dimensions.
        See np.gradient docs for more information.

    Returns
    -------
    K : np.ndarray.
        The result of Gaussian curvature of Z. Returns multidimensional
        array object with values of Gaussian curvature of shape `(n, n)`.

    """

    Zx, Zy = np.gradient(Z, h)
    Zxx, Zxy = np.gradient(Zx, h)
    _, Zyy = np.gradient(Zy, h)

    K = (Zxx * Zyy - (Zxy**2)) / (1 + (Zx**2) + (Zy**2)) ** 2

    return K