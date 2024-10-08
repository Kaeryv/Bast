from cmath import sqrt
import sys
from math import cos, sin
from typing import Tuple

import numpy as np

from scipy.linalg import norm

from khepri.constants import *

def nanometers(x: float) -> float:
    """ Convert nanometers to meters."""
    return x * 1e-9

def grid_center(shape: Tuple[int, int]) -> Tuple[int, int]:
    """Returns the central element of the reciprocal lattice with given shape."""
    assert shape[0] % 2 == shape[1] % 2, "Grid dimensions should have same parity."
    parity = shape[0] % 2 == 0
    if parity:
        return shape[0] // 2, shape[1] // 2
    else:
        return (shape[0] - 1) // 2, (shape[1] - 1) // 2


def grid_size(pw):
    assert pw[0] % 2 == pw[1] % 2, "Plane waves should have same parity."
    parity = pw[0] % 2 == 0
    if parity:
        nx, ny = pw[0] // 2, pw[1] // 2
        q = (4 * nx, 4 * ny)
    else:
        nx, ny = (pw[0] - 1) // 2, (pw[1] - 1) // 2
        q = (4 * nx + 1, 4 * ny + 1)
    
    return (nx, ny), q

def compute_kz(gx, gy, epsilon, wavelength, kp=[0.0, 0.0], dtype=np.complex128):
    kx, ky = kp  
    u = np.array([ kx + gx, ky + gy ], dtype)
    unorm = norm(u, axis=0)

    kz = np.emath.sqrt(epsilon * (twopi / wavelength)**2 - unorm**2)
    mask = np.logical_or(kz.imag < 0.0, np.logical_and(np.isclose(kz.imag, 0.0), kz.real < 0.0))
    np.negative(kz, where=mask, out=kz)
    return kz

def compute_eta(gx, gy, kp=[0.0, 0.0]):
    kx, ky = kp
    u = np.array([ kx + gx, ky + gy], dtype=np.complex128)
    unorm = norm(u, axis=0)
    mask = unorm**2 < np.finfo(float).eps
    not_mask = np.logical_not(mask)
    u[:, not_mask] = np.divide(u[:,not_mask], unorm[not_mask])
    u[0, mask], u[1, mask] = 1, 0
    #eta = np.cross(u, [0,0,1], axis=0)
    return u[1], -u[0]


def compute_mu(gx, gy, kzg, epsilon, wavelength, kp=[0.0, 0.0]):
    kx, ky = kp
    u = np.array([ kx + gx, ky + gy, np.zeros_like(gx) ], dtype=complex)
    unorm = norm(u, axis=0)
    mask = unorm**2 < np.finfo(float).eps
    not_mask = np.logical_not(mask)
    u[:, not_mask] = np.divide(u[:, not_mask], unorm[not_mask])
    u[:, mask] = np.array([[1,0,0]]).T
    mu = kzg / (sqrt(epsilon) * twopi  / wavelength) * u
    return mu[0], mu[1]

def coords_from_index(p: Tuple[int, int], n: Tuple[int, int], g: int):
    coord = np.unravel_index(g, p, order='F')
    return np.subtract(coord, n)

def epsilon_g(q, islands_data, eps_host, inverse=False):
    n = q[0] // 2, q[1] // 2
    eps_g = np.zeros(q, dtype=complex)
    trsf = (lambda x: x, lambda x: 1/x) [inverse]

    eps_g[n[0], n[1]] = trsf(eps_host)
    for bf, island_eps in islands_data:
        coef = trsf(island_eps) - trsf(eps_host)
        eps_g += coef * bf
    return eps_g

def incident(pw, E0=1.0, p_pol=0.5, s_pol=0.5):
    n = tuple(p // 2 for p in pw)
    nx, ny = n
    g0 = 0 + nx + pw[0] * (0 + ny)
    ng = pw[0] * pw[1]
    Nplus     = np.zeros(ng, dtype=np.complex128)
    Nplus[g0] = s_pol*E0
    Xplus     = np.zeros(ng, dtype=np.complex128)
    Xplus[g0] = p_pol*E0
    Nminus    = np.zeros(ng, dtype=np.complex128)
    Xminus    = np.zeros(ng, dtype=np.complex128)

    return np.asarray([ Nplus, Xplus, Nminus, Xminus ]).flatten()

def compute_currents(P_in, P_sca, lattice, wavelength, kp_inc=(0.,0.)):
    q = lattice.area / (2.0 * mu0c * twopi / wavelength)
    ng = P_in.shape[0] // 4
    rdtype = np.result_type(P_in, P_sca)

    j1plus  = np.zeros(ng, dtype=rdtype)
    j1minus = np.zeros(ng, dtype=rdtype)
    j3plus  = np.zeros(ng, dtype=rdtype)

    kgz_inc = lattice.kzi(wavelength, kp_inc).T.flat
    kgz_emerg = lattice.kze(wavelength, kp_inc).T.flat
    mask = np.abs(np.imag(kgz_inc)) <= sys.float_info.epsilon
    j1plus[mask] = q * kgz_inc[mask] * (np.abs(P_in[0:ng][mask])**2 + np.abs(P_in[ng:2*ng][mask])**2)
    j1minus[mask] = q * kgz_inc[mask]*(np.abs(P_sca[2*ng:3*ng][mask])**2+np.abs(P_sca[3*ng:4*ng][mask])**2)
    
    # Where substrate absorbs, transmission is zero
    if abs(np.imag(lattice.eps_emerg)) < sys.float_info.epsilon:
        mask = abs(np.imag(kgz_emerg)) < sys.float_info.epsilon
        j3plus[mask] = q * kgz_emerg[mask] * (np.abs(P_sca[0:ng][mask])**2 + np.abs(P_sca[ng:2*ng][mask])**2)
    
    return j1plus, j1minus, j3plus

def compute_fluxes(lattice, wavelength: float, pin, pout, kp: Tuple[float, float]= (0,0)) -> Tuple[float, float, float]:
    '''
        Parameters
        -------
        lattice : CartesianLattice
            A basis in which to express scattering matrices
        wavelength : float
            Excitation wavelength in meters
        Returns
        -------
        R_tot : float
            Normalized reflectance
        T_tot : float
            Normalized transmittance
        A_tot : float
            Absorption (obtained from R_tot and T_tot)
    '''
    j1plus, j1minus, j3plus = compute_currents(pin, pout, lattice, wavelength)
    R_tot=np.abs(np.sum(j1minus) / np.sum(j1plus))
    T_tot=np.abs(np.sum(j3plus) / np.sum(j1plus))

    return R_tot, T_tot, 1 - R_tot - T_tot
