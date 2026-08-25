import logging
from dataclasses import dataclass
import warnings


from .tools import compute_kplanar
from .alternative import (
    incident,
    poynting_fluxes,
    free_space_eigenmodes,
)
from .layer import Layer
from .expansion import Expansion

from .fields import (
    translate_mode_amplitudes,
    fourier_fields_from_mode_amplitudes,
    layer_eigenbasis_matrix,
)
from .fourier import idft
from .fields import longitudinal_fields
from .layer import stack_layers, stack_total

from .extension import ExtendedLayer as EL
from .operators import OperatorCapabilityError, solve_network_rt


from types import SimpleNamespace
from copy import copy
from cmath import sqrt as csqrt

import numpy as np
from numpy.linalg import solve
from time import perf_counter

from .misc import ensure_array


class FieldsNotRetainedWarning(UserWarning):
    """A field sample targeted a layer disabled by ``set_device``."""


@dataclass(frozen=True)
class FieldsNotRetained:
    """Return value for a field request in a non-retained device layer."""

    layer_name: str
    layer_index: int
    z: float

class Crystal:
    """This class has the goal to provide a simple interface for
    an end-user of the simulation code. For advanced usage, consider using the
    `Layer` API. Still, this class provides extensive functionality including:
    - Fields computation
    - Relfection / Transmission (total or split in diffraction orders)
    - Twisted PhCs management
    """
    def __init__(
        self, pw, lattice="square", lattice_pitch=1, void=False, epsi=1, epse=1
    ) -> None:
        """Creates a Crystal with a set number of plane waves in the expansion.
        Args:
            pw (tuple): The number of plane waves along x and y in the expansion.
            lattice (str or array): "square", "hexa" or explicit lattice vectors.
            lattice_pitch (str or array): when using "square" or "hexa" lattices.
            epsi (float): incidence medium epsilon
            epse (float): emergence medium epsilon
        """
        self.pw = pw
        self.a = lattice_pitch
        self.void = void
        self.epsi = epsi
        self.epse = epse
        self.source = None

        if isinstance(lattice, str):
            if lattice == "square":
                self.lattice = self.a * np.asarray(
                    [[1, 0], [0, 1]]
                )  # Each line is a lattice vector.
            elif lattice == "hexagonal":
                self.lattice = self.a * np.asarray(
                    [[np.sqrt(3) / 2, 0.5], [np.sqrt(3) / 2, -0.5]]
                )  # Each column is a lattice vector.
            else:
                raise NotImplementedError(
                    f"This {lattice} magic-string is not emplemented."
                )
        else:
            self.lattice = lattice

        self.expansion = Expansion(pw, self.lattice)

        self.layers = dict()
        self.stacking_matrices = list()
        self.stack_positions = []
        self._S = None
        self._Stot = None
        self._operator_solution = None
        self.solve_diagnostics = None
        self.internal_fields_requested = False
        self.device_fields_mask = ()

    @property
    def S(self):
        if self._S is None and self._operator_solution is not None:
            raise OperatorCapabilityError(
                "complete scattering matrix",
                "The operator backend solves only the configured incident "
                "source. Re-run solve(backend='dense') to obtain Crystal.S.",
            )
        return self._S

    @S.setter
    def S(self, value):
        self._S = value
        if value is not None:
            self._Stot = value

    @property
    def Stot(self):
        if self._Stot is None and self._operator_solution is not None:
            raise OperatorCapabilityError(
                "complete scattering matrix",
                "The operator backend solves only the configured incident "
                "source. Re-run solve(backend='dense') to obtain Crystal.Stot.",
            )
        return self._Stot

    @Stot.setter
    def Stot(self, value):
        self._Stot = value
        if value is not None:
            self._S = value

    @classmethod
    def from_expansion(cls, expansion, **kwargs):
        """Builds a Crystal from an existing expansion.
        Mostly useful for twisted expansion.
        Args:
            expansion (Expansion): The (twisted expansion)
            **kwargs: Passed to Crystal __init__
        Returns:
            The crystal with the specified expansion
        """
        obj = cls(expansion.pw, **kwargs)
        obj.expansion = expansion
        return obj

    def add_layer_uniform(self, name, epsilon, depth):
        """
        Add layer without planar structuration. It will be processed analytically
        without solving any eigenvalue problem. This is equivalent to a call to
        the more generic `add_layer` with a Layer object instancianted using `Layer.uniform`.
        """
        self.layers[name] = Layer.uniform(self.expansion, epsilon, depth)
    def add_pixmap_or_uniform(
        self,
        name,
        epsilon,
        depth,
        *,
        factorization="classical",
        normal_vectors=None,
    ):
        """
        Add layer from 2D ndarray that provides eps(x,y).
        """
        self.layers[name] = Layer.pixmap_or_uniform(
            self.expansion,
            epsilon,
            depth,
            factorization=factorization,
            normal_vectors=normal_vectors,
        )
    def add_layer_pixmap(
        self,
        name,
        epsilon,
        depth,
        *,
        factorization="classical",
        normal_vectors=None,
    ):
        """
        Add a layer from 2D ndarray that provides eps(x,y). This method will use FFT.
        """
        self.layers[name] = Layer.pixmap(
            self.expansion,
            epsilon,
            depth,
            factorization=factorization,
            normal_vectors=normal_vectors,
        )

    def add_layer_analytical(
        self,
        name,
        epsilon,
        epsilon_host,
        depth,
        *,
        factorization="classical",
        normal_vectors=None,
    ):
        """
        Add a layer from an islands description using analytical Fourier
        transforms.
        """
        self.layers[name] = Layer.analytical(
            self.expansion,
            epsilon,
            epsilon_host,
            depth,
            factorization=factorization,
            normal_vectors=normal_vectors,
        )

    def add_layer(self, name, layer, extended=False):
        if extended:
            self.layers[name] = EL(self.expansion, layer)
        else:
            self.layers[name] = layer

    def set_device(self, layers_stack, fields_mask=False):
        """
        Set the device stack and the layers whose internal fields are needed.

        ``fields_mask`` is the source of truth for device-layer field intent.
        When every entry is false, :meth:`solve` retains only the final
        scattering matrix.  A scalar boolean applies to every device layer.
        """
        self.device_stack = copy(layers_stack)
        self.global_stacking = []
        if not self.void:
            self.global_stacking.append("Sref")
        self.global_stacking.extend(layers_stack)
        if not self.void:
            self.global_stacking.append("Strans")

        required_layers = set(self.global_stacking)
        if "Sref" in required_layers and "Sref" not in self.layers:
            self.layers["Sref"] = Layer.half_infinite(
                self.expansion, "reflexion", self.epsi
            )
            self.layers["Sref"].fields = True
        if "Strans" in required_layers and "Strans" not in self.layers:
            self.layers["Strans"] = Layer.half_infinite(
                self.expansion, "transmission", self.epse
            )
            self.layers["Strans"].fields = True

        if isinstance(fields_mask, (bool, np.bool_)):
            normalized_mask = (bool(fields_mask),) * len(layers_stack)
        else:
            normalized_mask = tuple(fields_mask)
            if len(normalized_mask) != len(layers_stack):
                raise ValueError(
                    "fields_mask must contain one boolean per device layer"
                )
            if not all(
                isinstance(value, (bool, np.bool_))
                for value in normalized_mask
            ):
                raise TypeError("fields_mask values must be booleans")
            normalized_mask = tuple(bool(value) for value in normalized_mask)

        self.device_fields_mask = normalized_mask
        self.internal_fields_requested = any(normalized_mask)
        self.stack_retain_mask = list(normalized_mask)
        if not self.void:
            self.stack_retain_mask.insert(0, True)
            self.stack_retain_mask.append(True)

        # Repeated stack entries share one Layer eigenspace. Retain it if any
        # occurrence requests fields, while access remains occurrence-specific.
        retained_by_name = {
            name: name in ("Sref", "Strans")
            or any(
                enabled
                for stacked_name, enabled in zip(
                    self.global_stacking, self.stack_retain_mask
                )
                if stacked_name == name
            )
            for name in self.global_stacking
        }
        for name, enabled in retained_by_name.items():
            self.layers[name].fields = enabled
            if hasattr(self.layers[name], "base"):
                self.layers[name].base.fields = enabled

    @property
    def depth(self):
        depth = 0
        for name in self.device_stack:
            depth += self.layers[name].depth
        return depth
    @property
    def source_defined(self):
        return self.source is not None
    
    @property
    def solved(self):
        return self._S is not None or self._operator_solution is not None

    def solve(self, backend="dense", *, rtol=1e-10, atol=0.0, maxiter=500):
        """Solve the configured stack.

        Parameters
        ----------
        backend : {"dense", "operator", "auto"}
            ``dense`` preserves the historical full-S behavior. ``operator``
            computes source-specific reflected/transmitted amplitudes without
            assembling dense partial or total stack matrices. ``auto`` selects
            the operator path for compatible twisted stacks that do not retain
            internal fields, and otherwise uses ``dense``.
        rtol, atol, maxiter
            Iterative interface-solver controls used only by ``operator``.
        """
        assert self.source_defined, "Call set_source before solving."
        if backend not in ("auto", "dense", "operator"):
            raise ValueError("backend must be 'auto', 'dense', or 'operator'")
        stacked_layers = [self.layers[name] for name in self.global_stacking]
        operator_compatible = (
            not self.internal_fields_requested
            and any(isinstance(layer, EL) for layer in stacked_layers)
            and all(getattr(layer, "supports_operator", False) for layer in stacked_layers)
        )
        if backend == "auto":
            backend = "operator" if operator_compatible else "dense"
        if backend == "operator":
            if self.internal_fields_requested:
                requested = [
                    name
                    for name, enabled in zip(
                        self.device_stack, self.device_fields_mask
                    )
                    if enabled
                ]
                raise OperatorCapabilityError(
                    "internal fields",
                    "Fields were requested in device layer(s) "
                    f"{requested}. Set their fields_mask entries to False for "
                    "an R/T-only operator solve, or use backend='dense'.",
                )
            unsupported = [
                name
                for name, layer in zip(self.global_stacking, stacked_layers)
                if not getattr(layer, "supports_operator", False)
            ]
            if unsupported:
                raise OperatorCapabilityError(
                    "this layer stack",
                    "The following layers have no operator representation: "
                    + ", ".join(unsupported)
                    + ". Use backend='dense'.",
                )
            return self._solve_operator(stacked_layers, rtol, atol, maxiter)
        return self._solve_dense(stacked_layers)

    def _update_stack_positions(self, stacked_layers):
        layer_sizes = [layer.depth for layer in stacked_layers]
        self.stack_positions = list(np.cumsum(layer_sizes))
        self.stack_positions[-1] = np.inf
        if not self.void:
            self.stack_positions.insert(0, -np.inf)

    def _solve_dense(self, stacked_layers):
        started = perf_counter()
        self._operator_solution = None
        self._S = None
        self._Stot = None
        self.solve_diagnostics = None
        # Solving the required layers
        logging.debug('Solving each required layer')
        for name in dict.fromkeys(self.global_stacking):
            self.layers[name].solve(self.kp, self.source.wavelength)

        # The stack is built this way:
        # If reflexion side is on the left, the position is on the right, after the layer.
        # The S matrices vector corresponds to the matrix that starts from position
        # positions = [ -inf,  0,   d1,   d1 + d2  ]
        # Stot      = [ Sref,  S1,  S12,  S12T      ]
        # Srev      = [ S12T,  S2T,  ST,   I      ]
        self._update_stack_positions(stacked_layers)

        logging.debug('Building the layer stack')
        if self.internal_fields_requested:
            self.stacking_matrices, self.stacking_reverse_matrices, total = (
                stack_layers(
                    self.expansion.pw,
                    stacked_layers,
                    self.stack_retain_mask,
                )
            )
        else:
            total = stack_total(self.expansion.pw, stacked_layers)
            self.stacking_matrices = []
            self.stacking_reverse_matrices = []
        self._Stot = total
        self._S = total
        self.solve_diagnostics = {
            "backend": "dense",
            "iterations": 0,
            "residual": 0.0,
            "solve_seconds": perf_counter() - started,
            "operator_bytes": 0,
            "retained_bytes": total.nbytes,
            "estimated_dense_bytes": total.nbytes,
            "materialized_dense": True,
        }

    def _solve_operator(self, stacked_layers, rtol, atol, maxiter):
        started = perf_counter()
        self._S = None
        self._Stot = None
        self._operator_solution = None
        self.solve_diagnostics = None
        operators_by_name = {}
        for name in dict.fromkeys(self.global_stacking):
            operators_by_name[name] = self.layers[name].solve_operator(
                self.kp, self.source.wavelength
            )
        operators = [operators_by_name[name] for name in self.global_stacking]
        incident_amplitudes = incident(
            self.pw,
            self.source.te,
            self.source.tm,
            k_vector=(self.kp[0], self.kp[1], self.kzi),
        )
        operator_bytes = sum(operator.nbytes for operator in operators)
        estimated_dense_bytes = sum(
            operator.estimated_dense_bytes for operator in operators
        )
        try:
            solution = solve_network_rt(
                operators,
                incident_amplitudes,
                rtol=rtol,
                atol=atol,
                maxiter=maxiter,
            )
        finally:
            for layer in dict.fromkeys(stacked_layers):
                layer.operator = None
        self._operator_solution = solution
        self.stacking_matrices = []
        self.stacking_reverse_matrices = []
        self._update_stack_positions(stacked_layers)
        self.solve_diagnostics = {
            "backend": "operator",
            "iterations": solution.iterations,
            "residual": solution.residual,
            "solve_seconds": perf_counter() - started,
            "network_solve_seconds": solution.solve_seconds,
            "system_nnz": solution.system_nnz,
            "preconditioner_nnz": solution.preconditioner_nnz,
            "operator_bytes": operator_bytes,
            "retained_bytes": (
                solution.reflected.nbytes + solution.transmitted.nbytes
            ),
            "estimated_dense_bytes": estimated_dense_bytes,
            "materialized_dense": False,
        }
        return None

    def locate_layer(self, z):
        """Locates the layer at depth z and also computes the position relative to layer.

        Args:
            z (float): z depth

        Returns:
            tuple: The layer, the index and the relative depth.
        """
        assert z != np.nan
        layer_index = np.searchsorted(self.stack_positions, z) - 1
        layer_name = self.global_stacking[layer_index]
        if z <= 0:
            zr = z
        else:
            zr = z - self.stack_positions[layer_index]

        logging.debug(
            f"The fields position z = {z} is in layer {layer_index} named {layer_name}"
        )
        logging.debug(
            f"The layer goes from {self.stack_positions[layer_index]} to {self.stack_positions[layer_index+1]}"
        )
        logging.debug(f"zr={zr}")
        return self.layers[layer_name], layer_index, zr

    def _fields_not_retained(self, z, layer_index):
        layer_name = self.global_stacking[layer_index]
        result = FieldsNotRetained(layer_name, layer_index, float(z))
        warnings.warn(
            f"warning, asked fields in '{layer_name}' layer at z={z}, but "
            "its set_device fields_mask entry is False; returning "
            "FieldsNotRetained",
            FieldsNotRetainedWarning,
            stacklevel=3,
        )
        return result

    def _field_occurrence_retained(self, layer_index):
        # Exterior eigenspaces are already required for R/T. Their fields can
        # be reconstructed directly from Stot without partial stack matrices.
        if not self.void and layer_index in (0, len(self.global_stacking) - 1):
            return True
        return bool(self.stack_retain_mask[layer_index])
    
    def _fourier_fields(self, z, incident_fields):
        """Returns the fourier fields in the unit cell for a depth z.

        Args:
            z (float): z depth
            incident_fields (tuple): incident fields in Fourier space

        Returns:
            _type_: fourier fields at depth z.
        """
        if self._operator_solution is not None:
            raise OperatorCapabilityError(
                "electric or magnetic field samples",
                "The operator backend retains only reflected/transmitted modal "
                "amplitudes for the configured source. Request fields with "
                "set_device(..., fields_mask=...) and re-run "
                "solve(backend='dense').",
            )
        layer, layer_index, zr = self.locate_layer(z)
        if not self._field_occurrence_retained(layer_index):
            return self._fields_not_retained(z, layer_index)
        LI, WI, VI = layer.L, layer.W, layer.V
        RI = layer_eigenbasis_matrix(WI, VI)

        e = self.expansion
        Kx, Ky, _ = e.k_vectors(self.kp, self.source.wavelength)
        if isinstance(layer, Layer):
            W0, V0 = free_space_eigenmodes(Kx, Ky)
        else:
            W0, V0 = layer.W0, layer.V0
        R0 = layer_eigenbasis_matrix(W0, V0)

        k0 = 2 * np.pi / self.source.wavelength

        Wref = self.layers["Sref"].W
        Vref = self.layers["Sref"].V
        Rref = layer_eigenbasis_matrix(Wref, Vref)
        c1p = np.split(solve(Rref, incident_fields), 2)[0]
        c1m = self.Stot[0, 0] @ c1p
        c2p = self.Stot[1, 0] @ c1p
        if not self.void and layer_index == 0:
            cdplus, cdminus = c1p, c1m
        elif not self.void and layer_index == len(self.global_stacking) - 1:
            cdplus, cdminus = c2p, np.zeros_like(c2p)
        else:
            cdplus, cdminus = translate_mode_amplitudes(
                self.stacking_matrices[layer_index],
                self.stacking_reverse_matrices[layer_index],
                c1p,
                c1m,
                c2p,
            )
        d = layer.depth
        sx, sy, ux, uy = fourier_fields_from_mode_amplitudes(
            RI, LI, R0, (cdplus, cdminus), k0 * (d - zr)
        )

        sz, uz = longitudinal_fields((sx, sy, ux, uy), Kx, Ky, layer.IC)

        return sx, sy, sz, ux, uy, uz

    def get_source_as_field_vectors(self):
        """
        Construct a Field supervector from the current Crystal source.

        Returns:
            tuple: (E, H) in Fourier space
        """
        efield = incident(
            self.pw,
            self.source.te,
            self.source.tm,
            k_vector=(self.kp[0], self.kp[1], self.kzi),
        )
        hfield = np.zeros_like(efield)
        return efield, hfield

    def fields_coords_xy(self, x, y, z, incident_fields=None, kp=None, return_fourier=False):
        """Returns the fields at specified coordinates (x,y) for a depth z.

        Args:
            x (array): x coordinates
            y (array): y coordinates (must be len(x))
            z (scalar): z depth
            incident_fields (_type_): _description_

        Returns:
            tuple: The E and H fields, or :class:`FieldsNotRetained` when the
                target device-layer field marker is false.
        """
        x = ensure_array(x)
        y = ensure_array(y)
        assert x.shape == y.shape and (len(x.shape) == 2), "x and y must be 2D meshgrids"
        assert self.solved, "Call solve first."
        if self._operator_solution is not None:
            raise OperatorCapabilityError(
                "electric or magnetic field samples",
                "The operator backend retains only reflected/transmitted modal "
                "amplitudes for the configured source. Request fields with "
                "set_device(..., fields_mask=...) and re-run "
                "solve(backend='dense').",
            )
        if kp is None:
            kp = self.kp
        if incident_fields is None:
            incident_fields = np.hstack(self.get_source_as_field_vectors())
        elif isinstance(incident_fields, tuple) and len(incident_fields) == 2:
            incident_fields = np.hstack(incident_fields)
        e = self.expansion
        Kx, Ky, _ = e.k_vectors(kp, self.source.wavelength)

        if isinstance(z, str) and z == 'farfield':
            ffields = self._fourier_far_fields(incident_fields)
        else:
            ffields = self._fourier_fields(z, incident_fields)
            if isinstance(ffields, FieldsNotRetained):
                return ffields
        k0 = 2 * np.pi / self.source.wavelength
        
        if return_fourier:
            return ffields
        
        fields = [idft(s, k0 * Kx, k0 * Ky, x, y) for s in ffields]

        return np.split(np.asarray(fields), 2, axis=0)

    def fields_volume(self, x, y, z, incident_fields=None):
        """Return volume fields or the first disabled-layer request value."""

        z_values = tuple(z)
        for zi in z_values:
            _, layer_index, _ = self.locate_layer(zi)
            if not self._field_occurrence_retained(layer_index):
                return self._fields_not_retained(zi, layer_index)
        if incident_fields is None:
            incident_fields = self.get_source_as_field_vectors()
        
        fields = np.array([
            self.fields_coords_xy(x, y, zi, np.hstack(incident_fields))
            for zi in z_values
        ])
        return fields[:, 0, ...], fields[:, 1, ...]

    def set_source(self, wavelength, te=1.0, tm=1.0, theta=0.0, phi=0.0, kp=None):
        if kp is not None:
            self.kp = kxi, kyi = kp
            self.source = SimpleNamespace(
                te=te, tm=tm, theta=theta, phi=phi, wavelength=wavelength
            )
        else:
            self.source = SimpleNamespace(
                te=te, tm=tm, theta=theta, phi=phi, wavelength=wavelength
            )
            self.kp = kxi, kyi = compute_kplanar(
                self.epsi, wavelength, self.source.theta, self.source.phi
            )

        self.k0 = 2 * np.pi / wavelength
        self.kzi = np.conj(csqrt(self.k0**2 * self.epsi - kxi**2 - kyi**2))


    def poynting_flux_end(self, only_total=True):
        assert self.solved, "Call solve first"
        if self._operator_solution is not None:
            T = poynting_fluxes(
                self.expansion,
                self._operator_solution.transmitted,
                self.kp,
                self.source.wavelength,
                only_total=only_total,
                epsi=self.epsi,
                epse=self.epse,
            )
            R = poynting_fluxes(
                self.expansion,
                self._operator_solution.reflected,
                self.kp,
                self.source.wavelength,
                only_total=only_total,
                epsi=self.epsi,
                epse=self.epsi,
            )
            if only_total:
                return R.real, T.real
            return R, T
        incident_fields = incident(
            self.pw,
            self.source.te,
            self.source.tm,
            k_vector=(self.kp[0], self.kp[1], self.kzi),
        )
        Wref = self.layers["Sref"].W
        iWref = np.linalg.inv(Wref)
        c1p = iWref @ incident_fields  # [np.tile(lattice.trunctation,2)]
        Wtrans = self.layers["Strans"].W
        T = poynting_fluxes(
            self.expansion,
            Wtrans @ self.Stot[1, 0] @ c1p,
            self.kp,
            self.source.wavelength,
            only_total=only_total,
            epsi=self.epsi,
            epse=self.epse
        )
        R = poynting_fluxes(
            self.expansion,
            Wref @ self.Stot[0, 0] @ c1p,
            self.kp,
            self.source.wavelength,
            only_total=only_total,
            epsi=self.epsi,
            epse=self.epsi
        )
        if only_total:
            return R.real, T.real
        else:
            return R, T

    @property
    def zmax(self):
        return self.stack_positions[-2]



class Multilayer(Crystal):
    def __init__(self, epsi=1, epse=1):
        super().__init__((1,1), lattice="square", lattice_pitch=1, void=False, epsi=epsi, epse=epse)
