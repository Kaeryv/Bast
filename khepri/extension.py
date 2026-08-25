import logging
import numpy as np
from math import prod

from khepri.layer import Layer
from khepri.alternative import free_space_eigenmodes
from khepri.operators import ExtendedScatteringOperator, OperatorCapabilityError

def _embedding_indices(base_size, subspace_index, kind):
    """Indices occupied by one shifted base problem in the joint basis."""
    base_indices = np.arange(base_size)
    if kind == 0:
        return subspace_index * base_size + base_indices
    if kind == 1:
        return base_indices * base_size + subspace_index
    raise NotImplementedError("No more than two different lattices. Feel free to contribute!")


def _scatter_subspace(target, submatrix, subspace_index, kind=0):
    """Insert one ``2N x 2N`` matrix into a ``2N² x 2N²`` joint block."""
    base_size = submatrix.shape[0] // 2
    indices = _embedding_indices(base_size, subspace_index, kind)
    joint_size = base_size**2
    for output_polarization in range(2):
        source_rows = slice(
            output_polarization * base_size,
            (output_polarization + 1) * base_size,
        )
        target_rows = output_polarization * joint_size + indices
        for input_polarization in range(2):
            source_columns = slice(
                input_polarization * base_size,
                (input_polarization + 1) * base_size,
            )
            target_columns = input_polarization * joint_size + indices
            target[np.ix_(target_rows, target_columns)] = submatrix[
                source_rows, source_columns
            ]


def _scatter_scattering(target, scattering, subspace_index, kind=0):
    for output_port in range(2):
        for input_port in range(2):
            _scatter_subspace(
                target[output_port, input_port],
                scattering[output_port, input_port],
                subspace_index,
                kind,
            )


def _joint_subspace(submatrices: list, kind=0):
    """
    Join scattering matrices into larger common space.
    
    Parameters
    ----------
    submatrices: list
        Input submatrices shape=(2*N, 2*N)
    kind: int
        Merging mode
            0 -> matrices are most densely put in the extended one
            1 -> like an outer for loop
            2 -> left to implement, outer, outer loop for 3 different lattices

    Source
    ------
    Theory for Twisted Bilayer Photonic Crystal Slabs
    Beicheng Lou, Nathan Zhao, Momchil Minkov, Cheng Guo, Meir Orenstein, and Shanhui Fan
    https://doi.org/10.1103/PhysRevLett.126.136101
    """
    # N is the number of g vectors
    N = submatrices[0].shape[0] // 2
    result = np.zeros((2*N**2, 2*N**2), dtype=submatrices[0].dtype)
    for index, submatrix in enumerate(submatrices):
        _scatter_subspace(result, submatrix, index, kind)
    return result


def joint_subspace(submatrices: list, kind=0):
    '''
        Wrapper of _joint_subspace that processes 4 quadrants of smatrix
    '''
    base_size = submatrices[0].shape[-1] // 2
    dimension = 2 * base_size**2
    output = np.zeros((2, 2, dimension, dimension), dtype=submatrices[0].dtype)
    for index, scattering in enumerate(submatrices):
        _scatter_scattering(output, scattering, index, kind)
    return output


class ExtendedLayer():
    def __init__(self, expansion, base) -> None:
        if expansion.expansion_lhs == base.expansion:
            self.mode = 1
            self.gs = expansion.expansion_rhs.g_vectors
        elif expansion.expansion_rhs == base.expansion:
            self.mode = 0
            self.gs = expansion.expansion_lhs.g_vectors
        else:
            raise NotImplementedError(
                    "Base layer expansion should be in the extented expansion.")
        self.expansion = expansion
        self.base = base
        self.depth = self.base.depth
        self.fields = self.base.fields
        self._S = None
        self.operator = None
        self.operator_solved = False

    @property
    def S(self):
        if self._S is None and self.operator_solved:
            raise OperatorCapabilityError(
                "a dense extended-layer scattering matrix",
                "The compact shifted-subspace representation was selected. "
                "Solve the containing Crystal with backend='dense' if the full "
                "matrix is required.",
            )
        return self._S

    @S.setter
    def S(self, value):
        self._S = value

    @property
    def supports_operator(self):
        return not self.fields

    def solve_operator(self, k_parallel, wavelength):
        if self.fields:
            raise OperatorCapabilityError(
                "internal fields",
                "Disable this layer in set_device(..., fields_mask=...) or "
                "solve the containing Crystal with backend='dense'.",
            )
        base_scattering = None
        for shift_index, kp in enumerate(self.gs.T):
            self.base.fields = False
            self.base.solve(kp + k_parallel, wavelength)
            if base_scattering is None:
                base_scattering = np.empty(
                    (len(self.gs.T),) + self.base.S.shape,
                    dtype=self.base.S.dtype,
                )
            base_scattering[shift_index] = self.base.S
        self._S = None
        self.operator = ExtendedScatteringOperator(base_scattering, self.mode)
        self.operator_solved = True
        # The operator owns copies of every shifted base S matrix, including
        # the final one still referenced by ``base`` after the loop.
        self.base.S = None
        return self.operator
    
    def solve(self, k_parallel, wavelength):
        self.operator = None
        self.operator_solved = False
        WIs = list()
        VIs = list()
        LIs = list()
        if isinstance(self.base, Layer) and hasattr(self, "fields"):
            self.base.fields = self.fields

        extended_scattering = None
        for shift_index, kp in enumerate(self.gs.T):
            if isinstance(self.base, Layer):
                self.base.solve(kp + k_parallel, wavelength)
            else:
                self.base.set_source(wavelength, np.nan, np.nan, kp=kp + k_parallel)
                self.base.solve()

            if extended_scattering is None:
                base_size = self.base.S.shape[-1] // 2
                dimension = 2 * base_size**2
                extended_scattering = np.zeros(
                    (2, 2, dimension, dimension), dtype=self.base.S.dtype
                )
            _scatter_scattering(
                extended_scattering, self.base.S, shift_index, self.mode
            )
            if self.fields:
                WIs.append(self.base.W.copy())
                VIs.append(self.base.V.copy())
                LIs.append(np.diag(self.base.L).copy())

        mode = self.mode
        self.S = extended_scattering
        
        if self.fields:
            self.W = _joint_subspace(WIs, kind=mode)
            self.V = _joint_subspace(VIs, kind=mode)
            self.W0, self.V0 = extended_freespace(self.base.expansion, self.gs.T, wavelength, mode, k_parallel)
            self.L = np.diag(_joint_subspace(LIs, kind=mode))

        self.IC = 1.0

def extended_freespace(e, gs, wl, mode, k_parallel):
    V0s = list()
    W0s = list()
    for kp in gs:
        Kx0, Ky0, _ = e.k_vectors(kp + k_parallel, wl)
        W0, V0 = free_space_eigenmodes(Kx0, Ky0)
        W0s.append(W0)
        V0s.append(V0)

    W0 = _joint_subspace(W0s, kind=mode)
    V0 = _joint_subspace(V0s, kind=mode)

    return W0, V0
