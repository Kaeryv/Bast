"""Memory-efficient scattering operators and an R/T-only network solver."""

from dataclasses import dataclass
from inspect import signature
from time import perf_counter
from typing import Sequence

import numpy as np
from scipy.sparse import bmat, coo_matrix, csr_matrix, eye as sparse_eye
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.sparse.linalg import spilu


# SciPy renamed GMRES' relative tolerance from ``tol`` to ``rtol`` in 1.12.
# Python 3.8 resolves SciPy 1.10, while current Python versions resolve the
# newer API. Select the installed spelling once without weakening tolerances.
_GMRES_RTOL_KEY = "rtol" if "rtol" in signature(gmres).parameters else "tol"


class OperatorCapabilityError(RuntimeError):
    """Raised when a source-specific operator solve cannot provide a result."""

    def __init__(self, capability, guidance):
        self.capability = str(capability)
        self.guidance = str(guidance)
        super().__init__(
            f"operator backend cannot provide {self.capability}. {self.guidance}"
        )


class IterativeConvergenceError(RuntimeError):
    """Raised when the structured interface solve misses its tolerance."""

    def __init__(self, residual, iterations, dimension, estimated_dense_bytes):
        self.residual = float(residual)
        self.iterations = int(iterations)
        self.dimension = int(dimension)
        self.estimated_dense_bytes = int(estimated_dense_bytes)
        super().__init__(
            "structured scattering solve did not converge "
            f"(residual={self.residual:.3e}, iterations={self.iterations}, "
            f"dimension={self.dimension}, dense estimate="
            f"{self.estimated_dense_bytes / 2**30:.2f} GiB); use a smaller "
            "truncation or explicitly request backend='dense'"
        )


class DenseScatteringOperator:
    def __init__(self, scattering):
        self.scattering = np.asarray(scattering)
        self.dimension = self.scattering.shape[-1]
        self.dtype = self.scattering.dtype

    @property
    def nbytes(self):
        return self.scattering.nbytes

    @property
    def estimated_dense_bytes(self):
        return self.scattering.nbytes

    def block_matvec(self, output_port, input_port, vector):
        return self.scattering[output_port, input_port] @ vector

    def block_diagonal(self, output_port, input_port):
        return np.diag(self.scattering[output_port, input_port])

    def to_dense(self):
        return self.scattering.copy()

    def block_sparse(self, output_port, input_port):
        return csr_matrix(self.scattering[output_port, input_port])

    def clear_cache(self):
        pass


class HarmonicScatteringOperator:
    """Scattering matrix with independent 2-polarization harmonic blocks."""

    def __init__(self, harmonic_scattering):
        values = np.asarray(harmonic_scattering)
        if values.ndim != 5 or values.shape[1:] != (2, 2, 2, 2):
            raise ValueError("expected shape (harmonics, 2, 2, 2, 2)")
        self.harmonic_scattering = values
        self.harmonics = values.shape[0]
        self.dimension = 2 * self.harmonics
        self.dtype = values.dtype
        self._sparse_blocks = {}

    @property
    def nbytes(self):
        return self.harmonic_scattering.nbytes

    @property
    def estimated_dense_bytes(self):
        return 4 * self.dimension**2 * self.dtype.itemsize

    def block_matvec(self, output_port, input_port, vector):
        vector = np.asarray(vector).reshape(2, self.harmonics)
        blocks = self.harmonic_scattering[:, output_port, input_port]
        return np.einsum("hij,jh->ih", blocks, vector, optimize=True).reshape(-1)

    def block_diagonal(self, output_port, input_port):
        blocks = self.harmonic_scattering[:, output_port, input_port]
        return np.concatenate((blocks[:, 0, 0], blocks[:, 1, 1]))

    def to_dense(self):
        output = np.zeros(
            (2, 2, self.dimension, self.dimension), dtype=self.dtype
        )
        harmonic_indices = np.arange(self.harmonics)
        for output_port in range(2):
            for input_port in range(2):
                blocks = self.harmonic_scattering[:, output_port, input_port]
                for output_polarization in range(2):
                    rows = output_polarization * self.harmonics + harmonic_indices
                    for input_polarization in range(2):
                        columns = input_polarization * self.harmonics + harmonic_indices
                        output[
                            output_port, input_port, rows, columns
                        ] = blocks[:, output_polarization, input_polarization]
        return output

    def block_sparse(self, output_port, input_port):
        key = output_port, input_port
        if key not in self._sparse_blocks:
            blocks = self.harmonic_scattering[:, output_port, input_port]
            harmonics = np.arange(self.harmonics)
            rows = []
            columns = []
            data = []
            for output_polarization in range(2):
                for input_polarization in range(2):
                    rows.append(output_polarization * self.harmonics + harmonics)
                    columns.append(input_polarization * self.harmonics + harmonics)
                    data.append(blocks[:, output_polarization, input_polarization])
            self._sparse_blocks[key] = coo_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
                shape=(self.dimension, self.dimension),
            ).tocsr()
        return self._sparse_blocks[key]

    def clear_cache(self):
        self._sparse_blocks.clear()


class ExtendedScatteringOperator:
    """Joint-space scattering stored as shifted base matrices."""

    def __init__(self, base_scattering, kind=0):
        values = np.asarray(base_scattering)
        if values.ndim != 5 or values.shape[1:3] != (2, 2):
            raise ValueError("expected shape (shifts, 2, 2, 2N, 2N)")
        self.base_scattering = values
        self.kind = int(kind)
        self.base_size = values.shape[-1] // 2
        if values.shape[0] != self.base_size:
            raise ValueError("joint expansion currently requires equal base sizes")
        self.joint_harmonics = self.base_size**2
        self.dimension = 2 * self.joint_harmonics
        self.dtype = values.dtype
        self._indices = tuple(
            self._embedding_indices(index) for index in range(self.base_size)
        )
        self._sparse_blocks = {}

    def _embedding_indices(self, subspace_index):
        base = np.arange(self.base_size)
        if self.kind == 0:
            return subspace_index * self.base_size + base
        if self.kind == 1:
            return base * self.base_size + subspace_index
        raise NotImplementedError("only two joint-space embedding modes are supported")

    @property
    def nbytes(self):
        return self.base_scattering.nbytes + sum(index.nbytes for index in self._indices)

    @property
    def estimated_dense_bytes(self):
        return 4 * self.dimension**2 * self.dtype.itemsize

    def block_matvec(self, output_port, input_port, vector):
        vector = np.asarray(vector)
        output = np.empty(self.dimension, dtype=np.result_type(vector, self.dtype))
        for shift, indices in enumerate(self._indices):
            gathered = np.concatenate(
                (
                    vector[indices],
                    vector[self.joint_harmonics + indices],
                )
            )
            result = self.base_scattering[
                shift, output_port, input_port
            ] @ gathered
            output[indices] = result[: self.base_size]
            output[self.joint_harmonics + indices] = result[self.base_size :]
        return output

    def block_diagonal(self, output_port, input_port):
        output = np.empty(self.dimension, dtype=self.dtype)
        for shift, indices in enumerate(self._indices):
            diagonal = np.diag(
                self.base_scattering[shift, output_port, input_port]
            )
            output[indices] = diagonal[: self.base_size]
            output[self.joint_harmonics + indices] = diagonal[self.base_size :]
        return output

    def to_dense(self):
        output = np.zeros(
            (2, 2, self.dimension, self.dimension), dtype=self.dtype
        )
        for shift, indices in enumerate(self._indices):
            for output_port in range(2):
                for input_port in range(2):
                    base_block = self.base_scattering[
                        shift, output_port, input_port
                    ]
                    for output_polarization in range(2):
                        source_rows = slice(
                            output_polarization * self.base_size,
                            (output_polarization + 1) * self.base_size,
                        )
                        target_rows = (
                            output_polarization * self.joint_harmonics + indices
                        )
                        for input_polarization in range(2):
                            source_columns = slice(
                                input_polarization * self.base_size,
                                (input_polarization + 1) * self.base_size,
                            )
                            target_columns = (
                                input_polarization * self.joint_harmonics + indices
                            )
                            output[
                                output_port,
                                input_port,
                                target_rows[:, np.newaxis],
                                target_columns,
                            ] = base_block[source_rows, source_columns]
        return output

    def block_sparse(self, output_port, input_port):
        key = output_port, input_port
        if key not in self._sparse_blocks:
            rows = []
            columns = []
            data = []
            for shift, indices in enumerate(self._indices):
                joint_indices = np.concatenate(
                    (indices, self.joint_harmonics + indices)
                )
                rows.append(np.repeat(joint_indices, 2 * self.base_size))
                columns.append(np.tile(joint_indices, 2 * self.base_size))
                data.append(
                    self.base_scattering[
                        shift, output_port, input_port
                    ].reshape(-1)
                )
            self._sparse_blocks[key] = coo_matrix(
                (np.concatenate(data), (np.concatenate(rows), np.concatenate(columns))),
                shape=(self.dimension, self.dimension),
            ).tocsr()
        return self._sparse_blocks[key]

    def clear_cache(self):
        self._sparse_blocks.clear()


@dataclass
class NetworkSolution:
    reflected: np.ndarray
    transmitted: np.ndarray
    iterations: int
    residual: float
    solve_seconds: float
    system_nnz: int = 0
    preconditioner_nnz: int = 0


def _network_preconditioner(operators, interfaces, dimension, dtype):
    first_diagonal = []
    second_diagonal = []
    determinants = []
    for interface in range(interfaces):
        upper = operators[interface].block_diagonal(1, 1)
        lower = operators[interface + 1].block_diagonal(0, 0)
        determinant = 1 - upper * lower
        safe = np.where(np.abs(determinant) < 1e-14, 1.0, determinant)
        first_diagonal.append(upper)
        second_diagonal.append(lower)
        determinants.append(safe)

    def apply(vector):
        state = np.asarray(vector).reshape(interfaces, 2, dimension)
        output = np.empty_like(state, dtype=np.result_type(vector, dtype))
        for interface in range(interfaces):
            forward = state[interface, 0]
            backward = state[interface, 1]
            output[interface, 0] = (
                forward + first_diagonal[interface] * backward
            ) / determinants[interface]
            output[interface, 1] = (
                second_diagonal[interface] * forward + backward
            ) / determinants[interface]
        return output.reshape(-1)

    size = 2 * interfaces * dimension
    return LinearOperator((size, size), matvec=apply, dtype=dtype)


def solve_network_rt(
    operators: Sequence,
    incident,
    *,
    rtol=1e-10,
    atol=0.0,
    maxiter=500,
):
    """Solve a scattering network for one left-incident modal vector."""
    if not operators:
        raise ValueError("at least one scattering operator is required")
    dimension = operators[0].dimension
    if any(operator.dimension != dimension for operator in operators):
        raise ValueError("all scattering operators must have the same dimension")
    incident = np.asarray(incident)
    if incident.shape != (dimension,):
        raise ValueError(f"incident vector must have shape ({dimension},)")

    started = perf_counter()
    if len(operators) == 1:
        return NetworkSolution(
            operators[0].block_matvec(0, 0, incident),
            operators[0].block_matvec(1, 0, incident),
            0,
            0.0,
            perf_counter() - started,
        )

    interfaces = len(operators) - 1
    dtype = np.result_type(incident, *(operator.dtype for operator in operators))
    size = 2 * interfaces * dimension

    identity = sparse_eye(dimension, dtype=dtype, format="csr")
    blocks = [[None for _ in range(2 * interfaces)] for _ in range(2 * interfaces)]
    for interface in range(interfaces):
        forward_row = 2 * interface
        backward_row = forward_row + 1
        blocks[forward_row][forward_row] = identity
        blocks[forward_row][forward_row + 1] = -operators[
            interface
        ].block_sparse(1, 1)
        if interface > 0:
            blocks[forward_row][2 * (interface - 1)] = -operators[
                interface
            ].block_sparse(1, 0)

        blocks[backward_row][forward_row] = -operators[
            interface + 1
        ].block_sparse(0, 0)
        blocks[backward_row][backward_row] = identity
        if interface + 1 < interfaces:
            blocks[backward_row][2 * (interface + 1) + 1] = -operators[
                interface + 1
            ].block_sparse(0, 1)
    system = bmat(blocks, format="csr", dtype=dtype)
    rhs = np.zeros((interfaces, 2, dimension), dtype=dtype)
    rhs[0, 0] = operators[0].block_matvec(1, 0, incident)
    rhs = rhs.reshape(-1)
    preconditioner_nnz = 0
    try:
        incomplete_lu = spilu(
            system.tocsc(), drop_tol=1e-4, fill_factor=5, permc_spec="COLAMD"
        )
        preconditioner = LinearOperator(
            system.shape, matvec=incomplete_lu.solve, dtype=dtype
        )
        preconditioner_nnz = incomplete_lu.L.nnz + incomplete_lu.U.nnz
    except RuntimeError:
        preconditioner = _network_preconditioner(
            operators, interfaces, dimension, dtype
        )
    residual_history = []
    solution, info = gmres(
        system,
        rhs,
        M=preconditioner,
        atol=atol,
        restart=50,
        maxiter=maxiter,
        callback=residual_history.append,
        callback_type="pr_norm",
        **{_GMRES_RTOL_KEY: rtol},
    )
    rhs_norm = np.linalg.norm(rhs)
    residual = np.linalg.norm(system @ solution - rhs) / max(rhs_norm, 1.0)
    iterations = len(residual_history)
    estimated_dense_bytes = sum(
        operator.estimated_dense_bytes for operator in operators
    )
    normalized_tolerance = max(rtol, atol / max(rhs_norm, 1.0))
    if info != 0 or residual > normalized_tolerance:
        for operator in operators:
            operator.clear_cache()
        raise IterativeConvergenceError(
            residual, iterations, size, estimated_dense_bytes
        )

    system_nnz = system.nnz
    state = solution.reshape(interfaces, 2, dimension)
    reflected = operators[0].block_matvec(0, 0, incident)
    reflected += operators[0].block_matvec(0, 1, state[0, 1])
    transmitted = operators[-1].block_matvec(1, 0, state[-1, 0])
    for operator in operators:
        operator.clear_cache()
    return NetworkSolution(
        reflected,
        transmitted,
        iterations,
        residual,
        perf_counter() - started,
        system_nnz,
        preconditioner_nnz,
    )
