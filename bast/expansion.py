from enum import IntEnum
import numpy as np
from .tools import rotation_matrix


def generate_expansion_indices(pw):
    '''
        Returns expansion on square grid of reciprocal space.
        ndarray.shape = (2, prod(pw))
        Note: multiply by reciprocal lattice basis for @hex
    '''
    M = (pw[0] - 1) // 2
    N = (pw[1] - 1) // 2
    m = np.arange(-M, M+1)
    n = np.arange(-N, N+1)
    return np.vstack([e.flat for e in np.meshgrid(m, n)])

def kz_from_kplanar(kx, ky, k0, epsilon):
    arg = k0**2*epsilon-kx**2-ky**2
    kz = np.conj(np.sqrt(arg.astype("complex")))
    mask = np.logical_or(kz.imag < 0.0, np.logical_and(np.isclose(kz.imag, 0.0), kz.real < 0.0))
    np.negative(kz, where=mask, out=kz)
    return kz


class Truncation:
    NONE=0
    DISC=1

class Expansion:
    def __init__(self, pw, a=1.0) -> None:
        self.pw = pw
        self.a = a
        self.expansion_indices = generate_expansion_indices(pw)
        self._g_vectors = 2 * np.pi / a * self.expansion_indices
        self._k_vectors = None

    def _compute_k_vector(self, k_parallel, wavelength):
        self._k_vectors = np.zeros((3, np.prod(self.pw)), dtype=np.complex128)
        k0 = 2 * np.pi / wavelength
        self._k_vectors[0:2, :] = np.asarray(k_parallel)[:, np.newaxis] + self._g_vectors
        self._k_vectors[2, :] = kz_from_kplanar(*self._k_vectors[0:2, :], k0, 1.0)
        self._k_vectors /= k0




    def rotate(self, angle_deg):
        self._g_vectors = rotation_matrix(np.deg2rad(angle_deg)) @ self._g_vectors

    def __add__(self, rhs):
        '''
            Produces a new expansion that is the Minkowski sum of the two expansions self and rhs.
        '''
        g_sum = np.empty((2, self.g_vectors.shape[1]*rhs.g_vectors.shape[1]))
        i_sum = np.empty((2, self.g_vectors.shape[1]*rhs.g_vectors.shape[1]))
        i = 0
        for g2, i2 in zip(rhs.g_vectors.T, rhs.expansion_indices.T):
            for g1, i1 in zip(self.g_vectors.T, self.expansion_indices.T):
                g_sum[:, i] = g1 + g2
                i_sum[:, i] = i1 + i2
                i += 1
        pw_sum = (self.pw[0]**2, self.pw[1]**2)
        e = Expansion(pw_sum)
        e.expansion_indices = i_sum
        e._g_vectors = g_sum
        return e

    @property
    def g_vectors(self):
        return self._g_vectors.copy()

    def k_vectors(self, k_parallel, wavelength):
        self._compute_k_vector(k_parallel=k_parallel, wavelength=wavelength)
        return self._k_vectors
    
    def plot(self, what="gxy"):
        import matplotlib.pyplot as plt
        if what == "gxy":
            _, ax = plt.subplots()
            ax.plot(*self._g_vectors, 'r.')
            ax.axis("square")
            ax.axis("equal")
        elif what == "K":
            self._compute_k_vector((0,0), 0.3)
            ax = plt.figure().add_subplot(projection='3d')
            x = self._g_vectors[0].reshape(self.pw)
            y = self._g_vectors[1].reshape(self.pw)
            z = np.zeros(self.pw)
            ax.quiver(x*0,y*0,z*0, *self._k_vectors.reshape(3, *self.pw),arrow_length_ratio=0)
            ax.set_zlim(0, 1)
        plt.show()

    