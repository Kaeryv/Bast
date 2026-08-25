from enum import IntEnum
from khepri.tools import convolution_matrix, convolution_matrix_fourier
from khepri.fourier import transform, combine_fourier_masks
from khepri.alternative import (
    solve_structured_layer,
    solve_uniform_layer,
    build_scatmat,
    free_space_eigenmodes,
    scattering_reflection,
    scattering_transmission,
    scattering_identity,
    redheffer_product,
)
import numpy as np

from typing import Tuple

from .operators import (
    DenseScatteringOperator,
    HarmonicScatteringOperator,
    OperatorCapabilityError,
)


class Formulation(IntEnum):
    UNIFORM = 0
    FFT = 1
    ANALYTICAL = 2
    HALF_SPACE_INC = 3
    HALF_SPACE_TRN = 4


class Field(IntEnum):
    X = 0
    Y = 1
    Z = 2
    NORM = 3
    POYNTING = 4


def stack_layers(pw, layers, mask):
    '''
    Takes as input a list of layers and a conform list of booleans.
    The boolean dictates if we plan on keeping the fields for the layer.
    '''
    Stot = scattering_identity(pw, block=True)
    Sls = []
    for i, layer in enumerate(layers):
        Stot = redheffer_product(Stot, layer.S)
        if mask[i]:
            Sls.append(Stot.copy())
        else:
            Sls.append(None)

    Srev = scattering_identity(pw, block=True)
    mask = list(reversed(mask[1:]))
    Srs = []
    for i, layer in enumerate(reversed(layers[1:])):
        if mask[i]:
            Srs.append(Srev.copy())
        else:
            Srs.append(None)
        Srev = redheffer_product(layer.S.copy(), Srev)
    Srs.append(Srev.copy())
    Srs = list(reversed(Srs))
    return Sls, Srs, Stot


def stack_total(pw, layers):
    """Compose a stack while retaining only its rolling total S matrix."""
    total = scattering_identity(pw, block=True)
    for layer in layers:
        total = redheffer_product(total, layer.S)
    return total


class Layer:
    def __init__(self) -> None:
        self.formulation = None
        self.expansion = None

        self.W = None
        self.V = None
        self.L = None

        self._S = None
        self.operator = None
        self.operator_solved = False

        self.fields = False

    @property
    def S(self):
        if self._S is None and self.operator_solved:
            raise OperatorCapabilityError(
                "a dense per-layer scattering matrix",
                "Use the operator object's explicit to_dense() method before it "
                "is released, or solve the containing Crystal with "
                "backend='dense'.",
            )
        return self._S

    @S.setter
    def S(self, value):
        self._S = value

    @property
    def supports_operator(self):
        return self.formulation in (
            Formulation.HALF_SPACE_INC,
            Formulation.HALF_SPACE_TRN,
        ) or not self.fields

    def solve_operator(self, k_parallel, wavelength):
        """Return this layer through the most compact available operator.

        Uniform media and half spaces are represented as independent harmonic
        blocks. Patterned ordinary layers retain their usual dense per-layer
        RCWA matrix, but can still participate in the source-specific operator
        network without constructing dense partial or total stack matrices.
        """
        if not self.supports_operator:
            raise OperatorCapabilityError(
                "internal fields",
                "Disable this layer in set_device(..., fields_mask=...) or "
                "solve the containing Crystal with backend='dense'.",
            )
        if self.formulation in (Formulation.FFT, Formulation.ANALYTICAL):
            self.solve(k_parallel, wavelength)
            scattering = self._S
            self._S = None
            self.operator = DenseScatteringOperator(scattering)
            self.operator_solved = True
            return self.operator

        Kx, Ky, _ = self.expansion.k_vectors(k_parallel, wavelength)
        k0 = 2 * np.pi / wavelength
        harmonic_scattering = np.empty(
            (len(Kx), 2, 2, 2, 2), dtype=np.complex128
        )
        for harmonic, (kx, ky) in enumerate(zip(Kx, Ky)):
            kx = np.asarray([kx])
            ky = np.asarray([ky])
            W0, V0 = free_space_eigenmodes(kx, ky)
            if self.formulation == Formulation.UNIFORM:
                W, V, eigenvalues = solve_uniform_layer(kx, ky, self.epsilon)
                scattering = build_scatmat(
                    W, V, W0, V0, eigenvalues, self.depth, k0
                )
            elif self.formulation == Formulation.HALF_SPACE_INC:
                scattering, _, _, _ = scattering_reflection(
                    kx, ky, W0, V0, self.epsilon
                )
            else:
                scattering, _, _, _ = scattering_transmission(
                    kx, ky, W0, V0, self.epsilon
                )
            harmonic_scattering[harmonic] = scattering
        self._S = None
        self.operator = HarmonicScatteringOperator(harmonic_scattering)
        self.operator_solved = True
        return self.operator
    
    @classmethod
    def pixmap_or_uniform(cls, expansion, pixmap, depth):
        """
            This method is a convenience when you don't know what is inside pixmap.
            If the structure is rigorously uniform, it will be redirected to uniform solver.
            You better filter too small details before sending pixmap to this function.
            The expected use case is when doing optimization.
            Args:
                expansion (Expansion): The expansion to be used for this layer.
                pixmap (array): A numpy picture of you 2D pattern. Can be uniform.
                depth (float): The depth of the layer.
        """
        eps0 = pixmap.flatten()[0]
        if np.all(pixmap == eps0):
            return Layer.uniform(expansion, eps0, depth)
        else:
            return Layer.pixmap(expansion, pixmap, depth)


    @classmethod
    def pixmap(cls, expansion, pixmap, depth):
        """
            Constructing a layer this way will use the FFT algorithm to source the convolution matrix.
            The FFT will be applied on the real-space descritpion of the unit cell dielectric 'pixmap'.
            Args:
                expansion (Expansion): The expansion to be used for this layer.
                pixmap (array): A numpy picture of you 2D pattern.
                depth (float): The depth of the layer.
        """
        layer = cls()
        layer.expansion = expansion
        layer.formulation = Formulation.FFT
        layer.epsilon = pixmap
        layer.depth = depth
        return layer

    @classmethod
    def uniform(cls, expansion, epsilon, depth):
        layer = cls()
        layer.expansion = expansion
        layer.formulation = Formulation.UNIFORM
        layer.epsilon = epsilon
        layer.depth = depth
        return layer

    @classmethod
    def analytical(cls, expansion, islands_description, eps_host, depth):
        layer = cls()
        layer.expansion = expansion
        layer.formulation = Formulation.ANALYTICAL
        layer.epsilon = islands_description
        layer.eps_host = eps_host
        layer.depth = depth
        return layer

    @classmethod
    def half_infinite(cls, expansion, type, epsilon):
        layer = cls()
        if type == "reflexion":
            layer.formulation = Formulation.HALF_SPACE_INC
        elif type == "transmission":
            layer.formulation = Formulation.HALF_SPACE_TRN
        else:
            print("ERROR")
        layer.expansion = expansion
        layer.depth = 0
        layer.epsilon = epsilon
        return layer

    def solve(self, k_parallel: Tuple[float, float], wavelength: float):
        """
        Obtain the eigenspace and S-matrix from layer parameters.
        parameters:
        k_parallel: incident transverse wavevector.
        wavelength: excitation wavelength
        """
        self.operator = None
        self.operator_solved = False
        Kx, Ky, _ = self.expansion.k_vectors(k_parallel, wavelength)
        W0, V0 = free_space_eigenmodes(Kx, Ky)
        k0 = 2 * np.pi / wavelength

        if self.formulation == Formulation.FFT:
            self.C = convolution_matrix(self.epsilon, self.expansion.pw)
            self.IC = np.linalg.inv(self.C)
            self.W, self.V, self.L = solve_structured_layer(Kx, Ky, self.C)
            self.S = build_scatmat(self.W, self.V, W0, V0, self.L, self.depth, k0)
        if self.formulation == Formulation.ANALYTICAL:
            sigma = self.expansion.sigma
            Gx, Gy, epw = self.expansion.g_vectors_expanded(3)
            islands_data = [ (transform(isl["type"], isl["params"], Gx, Gy, sigma), isl["epsilon"]) for isl in self.epsilon ]

            
            fourier = combine_fourier_masks(islands_data, self.eps_host, inverse=False).T
            self.C = convolution_matrix_fourier(fourier.reshape(epw), self.expansion.pw)

            fourier = combine_fourier_masks(islands_data, self.eps_host, inverse=True).T
            self.IC = np.linalg.inv(self.C)

            self.W, self.V, self.L = solve_structured_layer(Kx, Ky, self.C)
            self.S = build_scatmat(self.W, self.V, W0, V0, self.L, self.depth, k0)
        elif self.formulation == Formulation.UNIFORM:
            self.W, self.V, self.L = solve_uniform_layer(Kx, Ky, self.epsilon)
            self.S = build_scatmat(self.W, self.V, W0, V0, self.L, self.depth, k0)
            self.IC = 1 / self.epsilon
        elif self.formulation == Formulation.HALF_SPACE_INC:
            self.S, self.W, self.V, self.L = scattering_reflection(
                Kx, Ky, W0, V0, self.epsilon
            )
            self.IC = 1 / self.epsilon
        elif self.formulation == Formulation.HALF_SPACE_TRN:
            self.S, self.W, self.V, self.L = scattering_transmission(
                Kx, Ky, W0, V0, self.epsilon
            )
            self.IC = 1 / self.epsilon

        if not self.fields:
            self.W = None
            self.V = None
            self.L = None
            self.IC = None
