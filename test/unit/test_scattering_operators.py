import unittest

import numpy as np
from numpy.testing import assert_allclose

from khepri.draw import Drawing
from khepri.expansion import Expansion
from khepri.extension import ExtendedLayer
from khepri.layer import Layer, stack_layers, stack_total
from khepri.operators import (
    DenseScatteringOperator,
    IterativeConvergenceError,
    OperatorCapabilityError,
    solve_network_rt,
)


class ScatteringOperatorTests(unittest.TestCase):
    def test_total_only_stack_matches_retained_stack(self):
        expansion = Expansion((3, 3))
        layers = [
            Layer.uniform(expansion, 1.5, 0.2),
            Layer.uniform(expansion, 2.25, 0.3),
            Layer.uniform(expansion, 1.2, 0.1),
        ]
        for layer in layers:
            layer.solve((0.13, -0.07), 1.4)
        _, _, retained_total = stack_layers(
            expansion.pw, layers, [True] * len(layers)
        )
        assert_allclose(stack_total(expansion.pw, layers), retained_total, atol=0.0)

    def test_uniform_harmonic_operator_materializes_exact_dense_matrix(self):
        expansion = Expansion((3, 3))
        layer = Layer.uniform(expansion, 2.25, 0.3)
        layer.solve((0.1, 0.2), 1.3)
        expected = layer.S.copy()
        operator = layer.solve_operator((0.1, 0.2), 1.3)
        assert_allclose(operator.to_dense(), expected, atol=2e-16)
        with self.assertRaises(OperatorCapabilityError):
            _ = layer.S

    def test_extended_operator_matches_both_dense_embeddings(self):
        pattern = Drawing((8, 8), 4.0)
        pattern.disc((0.0, 0.0), 0.25, 1.0)
        for base_side in (0, 1):
            with self.subTest(base_side=base_side):
                first = Expansion((3, 3))
                second = Expansion((3, 3))
                second.rotate(0.173)
                base_expansion = (second, first)[base_side]
                base = Layer.analytical(
                    base_expansion, pattern.islands(), 4.0, 0.2
                )
                layer = ExtendedLayer(first + second, base)
                layer.solve((0.0, 0.0), 1.25)
                expected = layer.S.copy()
                operator = layer.solve_operator((0.0, 0.0), 1.25)
                assert_allclose(operator.to_dense(), expected, atol=0.0)
                vector = np.linspace(0.0, 1.0, operator.dimension).astype(complex)
                vector += 1j * vector[::-1]
                for output_port in range(2):
                    for input_port in range(2):
                        assert_allclose(
                            operator.block_matvec(
                                output_port, input_port, vector
                            ),
                            expected[output_port, input_port] @ vector,
                            atol=5e-13,
                        )

    def test_inconsistent_network_raises_diagnostic_error(self):
        first = np.zeros((2, 2, 1, 1), dtype=complex)
        second = np.zeros_like(first)
        first[1, 0, 0, 0] = 1
        first[1, 1, 0, 0] = 1
        second[0, 0, 0, 0] = 1
        second[1, 0, 0, 0] = 1
        operators = [
            DenseScatteringOperator(first),
            DenseScatteringOperator(second),
        ]
        with self.assertRaises(IterativeConvergenceError) as raised:
            solve_network_rt(
                operators,
                np.ones(1, dtype=complex),
                maxiter=2,
            )
        self.assertGreater(raised.exception.residual, 1e-10)


if __name__ == "__main__":
    unittest.main()
