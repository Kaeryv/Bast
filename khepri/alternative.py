"""
    Alternative formulation of RCWA

    Work in progress, use with caution.

    Notations:
    S11 = S[0,0]
    S22 = S[1,1]
    S = [[S11, S12], [S21, S22]]
"""

import numpy as np
from numpy.lib.scimath import sqrt as csqrt
from numpy.linalg import inv, solve
from .tools import convolution_matrix
import logging
from math import prod

def redheffer_product(SA, SB):
    I = np.eye(SA[0,0].shape[0], dtype=np.complex128)
    D = I - SB[0,0] @ SA[1,1]
    F = I - SA[1,1] @ SB[0,0]

    S11 = SA[0, 0] + SA[0,1] @ solve(D, SB[0, 0]) @ SA[1, 0]
    S12 = SA[0,1] @ solve(D, SB[0, 1])
    S21 = SB[1,0] @ solve(F, SA[1, 0])
    S22 = SB[1, 1] + SB[1,0] @ solve(F, SA[1, 1]) @ SB[0, 1]
    
    S = np.array([[S11, S12], [S21, S22]])
    return S

def scattering_reflection(KX, KY, W0, V0, er, ur=1):
    N = len(KX)
    I = np.eye(KX.shape[0])
    KX = np.diag(KX)
    KY = np.diag(KY)

    Qref = np.vstack([
        np.hstack([KX @ KY,          er*ur*I - KX @ KX]),
        np.hstack([KY @ KY - er*ur*I,        - KY @ KX]),
    ]) / ur

    arg = (ur*er*I-KX**2-KY**2)
    arg = np.diag(arg)
    arg = arg.astype('complex')
    Kz = np.conj(csqrt(arg))
    eigenvals = np.hstack((1j*Kz, 1j*Kz))
    Wref = np.identity(2*N)
    Vref = Qref / eigenvals
    A = solve(W0, Wref) + solve(V0, Vref)
    B = solve(W0, Wref) - solve(V0, Vref)
    S11 = -solve(A, B)
    S12 = 2 * inv(A)
    S21 = 0.5 * (A - B @ solve(A, B))
    S22 = B @ inv(A)
    return np.array([[S11, S12], [S21, S22]]), Wref, Vref, eigenvals


def scattering_transmission(KX, KY, W0, V0, er, ur=1):
    N = len(KX)
    I = np.eye(KX.shape[0])
    KX = np.diag(KX)
    KY = np.diag(KY)
    Qref = np.vstack([
        np.hstack([KX @ KY,     er*ur*I - KX @ KX]),
        np.hstack([KY @ KY - er*ur*I,   - KY @ KX]),
    ]) / ur
    arg = (er*ur*I-KX**2-KY**2) #arg is kz^2
    arg = np.diag(arg)
    arg = arg.astype('complex')
    Kz = np.conj(csqrt(arg))
    eigenvals = np.hstack((1j*Kz, 1j*Kz))
    Wtrans = np.identity(2*N)

    Vtrans = Qref / eigenvals
    A = solve(W0, Wtrans) + solve(V0, Vtrans)
    B = solve(W0, Wtrans) - solve(V0, Vtrans)
    S11 = B @ inv(A) 
    S12 = 0.5 * (A - B @ solve(A, B))
    S21 = 2 * inv(A)
    S22 = - solve(A, B)
    return np.array([[S11, S12], [S21, S22]]), Wtrans, Vtrans, eigenvals

def free_space_eigenmodes(KX, KY):
    N = len(KX)
    KX = np.diag(KX)
    KY = np.diag(KY)
    I = np.identity(N)
    P = np.block([[KX*KY, I-KX**2], [KY**2-I, -KY*KX]])
    Q = P
    W = np.identity(2*N)
    arg = (I-KX**2-KY**2).astype('complex')
    kz = np.diag(arg).copy()
    mask = kz.real < 0
    kz[mask] = -1j * np.sqrt(-kz[mask])
    kz[~mask] = np.sqrt(kz[~mask])
    eigenvalues = np.hstack((1j*kz, 1j*kz))
    V = Q / eigenvalues 
    return W, V

def incident(pw, te_pol, tm_pol, k_vector, normalize=True):
    logging.debug(f"Building vector with plane wave {pw=}, {te_pol=}, {tm_pol=}, {k_vector=}")
    # Normalize in pol basis
    pol_norm = np.linalg.norm((abs(te_pol), abs(tm_pol)))
    if normalize:
        te_pol /= pol_norm
        tm_pol /= pol_norm

    kp = k_vector[0:2]
    kpnorm = np.linalg.norm(kp)
    knorm = np.linalg.norm(k_vector)
    kbar = np.array(k_vector) / knorm
    deviceNormalUnitVector = np.array([0, 0, -1], dtype=np.complex128)
    if abs(kpnorm) < 1e-8:
        aTE = np.array([1,0,0])
        aTM = np.array([0,1,0])
    else:
        aTE = - np.cross(deviceNormalUnitVector, kbar)
        aTE = aTE / np.linalg.norm(aTE)
        aTM = np.cross(aTE, kbar)
        aTM /= np.linalg.norm(aTM)

    logging.debug(f"{aTM=}, {aTE=}, {kbar=}, {knorm=}")
    N = prod(pw)
    delta = np.zeros(N, dtype=np.complex128)
    delta[(N-1)//2] = 1
    pxy = te_pol * aTE + tm_pol * aTM
    return np.hstack([delta*pxy[0], delta*pxy[1]])

def solve_uniform_layer(Kx, Ky, er, m_r = 1):
    '''
        Computes P & Q matrices for homogeneous layer.
    '''
    Kx = np.diag(Kx)
    Ky = np.diag(Ky)
    N = len(Kx)
    I = np.identity(N, dtype=np.complex128)
    P = (1/er) * np.block(
        [
            [ Kx * Ky,              er * m_r * I - Kx**2 ], 
            [ Ky**2 - m_r * er * I, - Ky * Kx ]
        ])
    Q = (er/m_r) * P
    W = np.identity(2*N)
    arg = (m_r*er*I-Kx**2-Ky**2)
    kz = np.diag(arg).astype('complex')
    mask = kz.real < 0
    kz[mask] = -1j * np.sqrt(-kz[mask])
    kz[~mask] = np.sqrt(kz[~mask])
    #Kz = np.conj(csqrt(np.diag(arg)))
    eigenvalues = np.hstack((1j*kz, 1j*kz))
    #mask = np.logical_or(eigenvalues.imag < 0.0, np.logical_and(np.isclose(eigenvalues.imag, 0.0), eigenvalues.real < 0.0))
    #np.negative(eigenvalues, where=mask, out=eigenvalues)
    V = Q / eigenvalues

    return W, V, eigenvalues

def normal_vector_permittivity(
    epsilon_convolution, reciprocal_convolution, normal_convolutions
):
    """Return the in-plane permittivity operators for Li factorization.

    ``normal_convolutions`` contains the convolution matrices of
    :math:`n_x^2`, :math:`n_x n_y`, and :math:`n_y^2` for a real unit-normal
    field continued periodically over the unit cell.  Products normal to a
    material boundary use Li's inverse rule, while tangential products retain
    Laurent's rule.
    """
    epsilon_convolution = np.asarray(epsilon_convolution, dtype=np.complex128)
    reciprocal_convolution = np.asarray(
        reciprocal_convolution, dtype=np.complex128
    )
    if epsilon_convolution.ndim != 2 or (
        epsilon_convolution.shape[0] != epsilon_convolution.shape[1]
    ):
        raise ValueError("epsilon convolution matrix must be square")
    if reciprocal_convolution.shape != epsilon_convolution.shape:
        raise ValueError("reciprocal convolution matrix has the wrong shape")
    if len(normal_convolutions) != 3:
        raise ValueError("normal_convolutions must contain nxx, nxy, and nyy")
    nxx, nxy, nyy = tuple(
        np.asarray(matrix, dtype=np.complex128)
        for matrix in normal_convolutions
    )
    if any(
        matrix.shape != epsilon_convolution.shape
        for matrix in (nxx, nxy, nyy)
    ):
        raise ValueError("normal convolution matrices have the wrong shape")

    identity = np.eye(epsilon_convolution.shape[0], dtype=np.complex128)
    inverse_rule = solve(reciprocal_convolution, identity)
    discontinuity = epsilon_convolution - inverse_rule
    epsilon_xx = epsilon_convolution - discontinuity @ nxx
    epsilon_xy = -discontinuity @ nxy
    epsilon_yy = epsilon_convolution - discontinuity @ nyy
    return epsilon_xx, epsilon_xy, epsilon_yy


def solve_structured_layer(kx, ky, C, factorized_permittivity=None):
    KX = np.diag(kx)
    KY = np.diag(ky)
    
    I = np.eye(KX.shape[0], dtype=np.longdouble)
    Pi = np.vstack([
        np.hstack([KX @ solve(C, KY),     I - KX @ solve(C, KX)]),
        np.hstack([KY @ solve(C, KY) - I,   - KY @ solve(C, KX)]),
    ]).astype(np.complex128)
    if factorized_permittivity is None:
        Qi = np.vstack([
            np.hstack([KX @ KY,     C - KX @ KX]),
            np.hstack([KY @ KY - C,   - KY @ KX]),
        ]).astype(np.complex128)
    else:
        epsilon_xx, epsilon_xy, epsilon_yy = tuple(
            np.asarray(matrix, dtype=np.complex128)
            for matrix in factorized_permittivity
        )
        if any(
            matrix.shape != C.shape
            for matrix in (epsilon_xx, epsilon_xy, epsilon_yy)
        ):
            raise ValueError(
                "factorized permittivity matrices have the wrong shape"
            )
        Qi = np.vstack([
            np.hstack([KX @ KY + epsilon_xy, epsilon_yy - KX @ KX]),
            np.hstack([KY @ KY - epsilon_xx, -KY @ KX - epsilon_xy]),
        ]).astype(np.complex128)
    
    lam2i, WI = np.linalg.eig(Pi @ Qi)
    lam = np.sqrt(lam2i+0j) # changed from np.sqrt()
    #lam = np.where(np.imag(lam) < 0, -lam, lam)
    #mask = np.logical_or(lam.imag < 0.0, np.logical_and(np.isclose(lam.imag, 0.0), lam.real < 0.0))
    #np.negative(lam, where=mask, out=lam)
    VI = Qi @ WI / lam
    return WI, VI, lam


def build_scatmat(WI, VI, W0, V0, lambdas, dbar, k0):
    t1 = solve(WI, W0)
    t2 = solve(VI, V0)
    A = t1 + t2
    B = t1 - t2
    X = np.diag(np.exp(-lambdas*dbar*k0))

    # Build the transfer matrix
    T = A - X @ B @ solve(A, X) @ B

    # Convert to scattering matrix
    S11 = solve(T , ( X @ B @ solve(A, X) @ A - B))
    S12 = solve(T , X @ (A - B @ solve(A, B)))
    S21, S22 = S12, S11
    return np.array([[S11, S12], [S21, S22]])

def scattering_uniform_layer(expansion, kp, wl, eps_layer, depth, return_eigenspace=False):
    kx, ky, _ = expansion.k_vectors(kp, wl)
    W0, V0 = free_space_eigenmodes(kx, ky)
    k0 = 2 * np.pi / wl
    WI, VI, LI =  solve_uniform_layer(kx, ky, eps_layer)
    S = build_scatmat(WI, VI, W0, V0, LI, depth, k0)
    if return_eigenspace:
        return S, (LI, WI, VI)
    else:
        return S

def scattering_structured_layer(expansion, kp, wl, epsilon_map, depth, return_eigenspace=False):
    kx, ky, _ = expansion.k_vectors(kp, wl)
    W0, V0 = free_space_eigenmodes(kx, ky)
    k0 = 2 * np.pi / wl
    C = convolution_matrix(epsilon_map, expansion.pw)
    WI, VI, LI =  solve_structured_layer(kx, ky, C)
    S = build_scatmat(WI, VI, W0, V0, LI, depth, k0)
    if return_eigenspace:
        return S, (LI, WI, VI)
    else:
        return S

def scattering_identity(pw, block=False):
    I = np.eye(2*prod(pw))
    if block:
        SI = SI = np.asarray([
            [np.zeros_like(I), I],
            [I, np.zeros_like(I)],
        ]).astype('complex')
    else:
        SI = np.vstack([
            np.hstack([np.zeros_like(I), I]),
            np.hstack([I, np.zeros_like(I)]),
        ]).astype('complex')
    return SI


def poynting_fluxes(expansion, c_output, kp, wavelength, only_total=True, epsi=1, epse=1):
    k0 = 2 * np.pi / wavelength
    kzi = np.conj(csqrt(k0**2*epsi-kp[0]**2-kp[1]**2)) / k0
    sx, sy = np.split(c_output, 2)
    kx, ky, kz = expansion.k_vectors(kp, wavelength, epse)
    sz = - (kx * sx + ky * sy) / kz
    t = kz.real / np.real(kzi) * (np.abs(sx)**2+np.abs(sy)**2+np.abs(sz)**2)
    if only_total:
        return np.sum(t)
    else:
        return np.sum(t), t
