import numpy as np

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