"""
    Dielectric veins as in Molding The Flow of Light
"""

import unittest

from khepri.tmat.tools import nanometers
from khepri.tmat.matrices import multS
from khepri.tmat.lattice import CartesianLattice
import numpy as np
from scipy.io import loadmat
from numpy.testing import assert_allclose
from khepri.tmat.tools import coords_from_index

from khepri.tmat.tools import epsilon_g, grid_size
from khepri.fourier import transform

fixtures = "./test/integration/fixtures/veins/"
pw = (6, 6)
a  = nanometers(100)
wavelength = nanometers(200)
kp = (0, 0)
veins = 0.165 / 2

class TestVeins(unittest.TestCase):
    def test_lattice(self):
        lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0, eps_incid=1.0, dtype=np.float64)
        data = loadmat(f"{fixtures}/reciproc.mat")
        self.assertTrue(np.allclose(lattice.b1, data['b1']))
        self.assertTrue(np.allclose(lattice.b2, data['b2']))

        data = loadmat(f"{fixtures}/gvectors.mat")
        self.assertTrue(np.allclose(lattice.gx, data['gx']))
        self.assertTrue(np.allclose(lattice.gy, data['gy']))

        self.assertFalse(np.allclose(lattice.gy, data['gx']))

        # Big grid
        self.assertTrue(np.allclose(lattice.Gx, data['Gx']))
        self.assertTrue(np.allclose(lattice.Gy, data['Gy']))


    def test_fourier(self):
        from khepri.fourier import transform
        # Unwrap MATLAB's 1x1 complex cell values explicitly for NumPy 2.
        data = loadmat(f"{fixtures}/fourier.mat", squeeze_me=True)
        val = np.asarray(data['Omega_g'], dtype=complex)
        
        lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0, eps_incid=1.0, dtype=np.float64)
        boolean_field = transform("rectangle", [veins * a, veins*a, a*(1-veins), a  *(1-veins) ], lattice.Gx, lattice.Gy, lattice.area)
        
        assert_allclose(boolean_field, val)

        self.assertFalse(np.allclose(boolean_field.real, val.imag))

    def test_kz(self):
        data = loadmat(f"{fixtures}/kz.mat")
        lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0, eps_incid=1.0, dtype=np.float64)
        kzi = lattice.kzi(wavelength, kp)
        kze = lattice.kze(wavelength, kp)
        assert_allclose(kzi, data["kgz_in"])
        assert_allclose(kze, data["kgz_em"])

        # TODO: Fix imaginary parts 0.0 - > 1.0j in matlab strange behaviour
        # When a = lambda (quite normal)



        

    def test_polarization_basis(self):
        data = loadmat(f"{fixtures}/eta.mat")
        lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0, eps_incid=1.0, dtype=np.float64)
        eta = lattice.eta(wavelength, kp)
        assert_allclose(eta[0], data["etagx"])
        assert_allclose(eta[1], data["etagy"])

        data = loadmat(f"{fixtures}/mu.mat")
        # print(data)
        mugi = lattice.mugi(wavelength, kp)
        assert_allclose(mugi[0], data["mugx_in"], atol=1e-7)
        assert_allclose(mugi[1], data["mugy_in"], atol=1e-7)

        U, Vi = lattice.U(wavelength, kp), lattice.Vi(wavelength, kp)
        data = loadmat(f"{fixtures}/UV.mat")
        assert_allclose(U, data["U_in"])
        assert_allclose(Vi, data["V_in"])

        lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0, eps_incid=1.0, dtype=np.float64)
        boolean_field = transform("rectangle", [veins * a, veins*a, a*(1-veins), a  *(1-veins) ], lattice.Gx, lattice.Gy, lattice.area)
        
        data = loadmat(f"{fixtures}/epsg.mat")
        _, q = grid_size(pw)
        epsg = epsilon_g(q, [(boolean_field, 1.0)], 8.9)
        iepsg = epsilon_g(q, [(boolean_field, 1.0)], 8.9, inverse=True)
        assert_allclose(epsg, data["eps_g"])
        assert_allclose(iepsg, data["epsinv_g"])

    def test_matrices(self):
        lattice = CartesianLattice(pw, a1=(a, 0.0), a2=(0.0, a), eps_emerg=1.0, eps_incid=1.0, dtype=np.float64)
        U, Vi = lattice.U(wavelength, kp), lattice.Vi(wavelength, kp)
        
        boolean_field = transform("rectangle", [veins * a, veins*a, a*(1-veins), a  *(1-veins) ],  lattice.Gx, lattice.Gy, lattice.area)
        data = loadmat(f"{fixtures}/S.mat")
        
        from khepri.tmat.matrices import matrix_a, matrix_s
        from scipy.linalg import expm
        _, q = grid_size(pw)
        epsg = epsilon_g(q, [(boolean_field, 1.0)], 8.9)
        iepsg = epsilon_g(q, [(boolean_field, 1.0)], 8.9, inverse=True)
        nx, ny = lattice.gx.shape[0] // 2, lattice.gx.shape[1] // 2
        indices = np.array([ coords_from_index(pw, (nx,ny), i) for i in range(pw[0] * pw[1]) ])
        A = matrix_a(indices, lattice.gx, lattice.gy, epsg, iepsg, wavelength, kx=kp[0], ky=kp[1])
        assert_allclose(A, data["A"])

        data2 = loadmat(f"{fixtures}/T.mat")
        slice_depth = 1e-7 / float(2**3)
        Ad = - A * slice_depth
        assert_allclose(Ad, data2["Ad"])
        Txyz = expm(-A * slice_depth)
        assert_allclose(Txyz, data2["T_xyz"], atol=1e-13)
        T = U @ Txyz @ Vi # Must be kept, part of the definition
        assert_allclose(T, data["Tslice"], atol=1e-11)
        S = matrix_s(T)
        for _ in range(3):
            S = multS(S, S)
        assert_allclose(S, data["Sslice"], atol=1e-13)

if __name__ == '__main__':
    unittest.main()
