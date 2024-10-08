from .tools import coords_from_index
from .matrices import *
from .lattice import complex_dtype
from scipy.linalg import expm
from ..fourier import transform
from khepri.tmat.tools import grid_size, epsilon_g


class LayerMatrix():
    def __init__(self, S, keep_metadata=False):
        self.S = S

    def numpy(self):
        return self.S
    
    def copy(self):
        return LayerMatrix(self.S.copy())

    def __matmul__(self, rhs):
        if isinstance(rhs, LayerMatrix):
            return LayerMatrix(multS(self.S, rhs.S))
        else:
            return self.S @ rhs

def scattering_matrix(pw, lattice, island_type, params, island_eps, eps_host, wavelength, kp, depth=100e-9, slicing_pow=3):
    _, q = grid_size(pw)
    gx, gy = lattice.gx, lattice.gy
    boolean_field = transform(island_type, params, lattice.Gx, lattice.Gy, lattice.area)
    U = lattice.U(wavelength, kp)
    Vi =lattice.Vi(wavelength, kp)

    epsg = epsilon_g(
            q, [(boolean_field, island_eps)], eps_host
    )
    epsinvg = epsilon_g(
            q, [(boolean_field, island_eps)], eps_host, 
            inverse=True
    )
    nx, ny = gx.shape[0] // 2, gx.shape[1] // 2
    indices = np.array([ coords_from_index(pw, (nx,ny), i) for i in range(pw[0] * pw[1]) ])
    A = matrix_a(indices, gx, gy, epsg, epsinvg, wavelength, kx=kp[0], ky=kp[1], dtype=complex_dtype[lattice.dtype])
    slice_depth = lattice.dtype(depth) / lattice.dtype(2**slicing_pow)
    T = U @ expm(-A * slice_depth) @ Vi
    
    S = matrix_s(T)
    for _ in range(slicing_pow):
        S1 = S.copy()
        S = multS(S1, S1)
        del S1

    return LayerMatrix(S)


def scattering_matrix_islands(pw, lattice, islands_description, eps_host, wavelength, kp, depth=100e-9, slicing_pow=3):
    _, q = grid_size(pw)
    gx, gy = lattice.gx, lattice.gy
    islands_data = [ (transform(isl["shape"], isl["params"], lattice.Gx, lattice.Gy, lattice.area), isl["epsilon"]) for isl in islands_description ]

    epsg = epsilon_g(
            q, islands_data, eps_host
    )
    epsinvg = epsilon_g(
            q, islands_data, eps_host, 
            inverse=True
    )

    U = lattice.U(wavelength, kp)
    Vi =lattice.Vi(wavelength, kp)
    nx, ny = gx.shape[0] // 2, gx.shape[1] // 2
    indices = np.array([ coords_from_index(pw, (nx,ny), i) for i in range(pw[0] * pw[1]) ])
    A = matrix_a(indices, gx, gy, epsg, epsinvg, wavelength, kx=kp[0], ky=kp[1], dtype=complex_dtype[lattice.dtype])
    slice_depth = lattice.dtype(depth) / lattice.dtype(2**slicing_pow)
    T = U @ expm(-A * slice_depth) @ Vi
    
    S = matrix_s(T)
    for _ in range(slicing_pow):
        S1 = S.copy()
        S = multS(S1, S1)
        del S1

    return LayerMatrix(S)


def scattering_air_tmp(pw, lattice, wavelength, depth):
    ng = prod(pw)
    kzs = compute_kz(lattice.gx, lattice.gy, 1.0, wavelength)
    S = np.zeros((4*ng, 4*ng), dtype=complex_dtype[lattice.dtype])
    
    for i, kz in enumerate(kzs.flat):
        for j in range(4):
            S[i+j*ng, i+j*ng] = np.exp(1j*kz*depth)
    return LayerMatrix(S)



def scattering_matrix_npy(pw, lattice, island_data, island_eps, eps_host, wavelength, theta_deg=0.0, phi_deg=0.0, depth=100e-9, slicing_pow=3):
    nx, ny = pw[0] // 2, pw[1] // 2
    q = (4 * nx, 4 * ny)
    gx, gy = lattice.gx, lattice.gy
    boolean_field = island_data.copy()
    assert boolean_field.shape == lattice.Gx.shape
    kp = lattice.kp(wavelength, theta_deg, phi_deg)
    U, Vi = lattice.U(wavelength, theta_deg, phi_deg), lattice.Vi(wavelength, theta_deg, phi_deg)
    epsg = epsilon_g(
            q, [boolean_field], [island_eps], eps_host
    )
    epsinvg = epsilon_g(
            q, [boolean_field], [island_eps], eps_host, 
            inverse=True
    )
    
    A = matrix_a(gx, gy, epsg, epsinvg, wavelength, kx=kp[0], ky=kp[1])
    slice_depth = depth / float(2**slicing_pow)
    T = U @ expm(-A * slice_depth) @ Vi
    S = matrix_s(T)

    for i in range(slicing_pow):
        S1 = S.copy()
        S = multS(S1,S1)
        del S1
    return LayerMatrix(S)

def scattering_identity(pw):
    I = np.eye(2*prod(pw))
    SI = np.vstack([
        np.hstack([np.zeros_like(I), I]),
        np.hstack([I, np.zeros_like(I)]),
    ])
    return SI
def scattering_interface(lattice, wavelength, kp=(0,0)):
    U = lattice.U(wavelength, kp=kp)
    Ve = lattice.Ve(wavelength, kp=kp)
    T_interface = U @ Ve
    return LayerMatrix(matrix_s(T_interface))
