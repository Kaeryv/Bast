import unittest

import numpy as np

from khepri.alternative import normal_vector_permittivity
from khepri.crystal import Crystal
from khepri.expansion import Expansion
from khepri.layer import Layer
from khepri.tools import convolution_matrix


class FourierFactorizationTests(unittest.TestCase):
    def test_lamellar_normal_reduces_to_inverse_rule(self):
        epsilon_map = np.ones((64, 3), dtype=np.complex128)
        epsilon_map[:27] = (3.18 - 4.41j) ** 2
        harmonics = (5, 1)
        epsilon = convolution_matrix(epsilon_map, harmonics)
        reciprocal = convolution_matrix(1 / epsilon_map, harmonics)
        identity = np.eye(5, dtype=np.complex128)

        epsilon_xx, epsilon_xy, epsilon_yy = normal_vector_permittivity(
            epsilon, reciprocal, (identity, 0 * identity, 0 * identity)
        )

        np.testing.assert_allclose(
            epsilon_xx, np.linalg.inv(reciprocal), atol=2e-12
        )
        np.testing.assert_allclose(epsilon_xy, 0.0, atol=2e-12)
        np.testing.assert_allclose(epsilon_yy, epsilon, atol=2e-12)

    def test_normal_vector_api_is_opt_in_and_backward_compatible(self):
        expansion = Expansion((3, 1))
        epsilon = np.ones((8, 3))
        epsilon[:4] = 2.25

        classical = Layer.pixmap(expansion, epsilon, 0.2)
        self.assertEqual(classical.factorization, "classical")
        with self.assertRaisesRegex(ValueError, "requires normal_vectors"):
            Layer.pixmap(
                expansion, epsilon, 0.2, factorization="normal-vector"
            )
        with self.assertRaisesRegex(ValueError, "require factorization"):
            Layer.pixmap(expansion, epsilon, 0.2, normal_vectors=(1, 0))

    def test_constant_normal_is_normalized_and_matches_sampled_field(self):
        expansion = Expansion((3, 1))
        epsilon = np.ones((16, 3))
        epsilon[:7] = 4.0
        constant = Layer.pixmap(
            expansion,
            epsilon,
            0.2,
            factorization="normal-vector",
            normal_vectors=(7.0, 0.0),
        )
        sampled = Layer.pixmap(
            expansion,
            epsilon,
            0.2,
            factorization="normal-vector",
            normal_vectors=(np.ones_like(epsilon), np.zeros_like(epsilon)),
        )
        _, constant_factorized = constant._patterned_fourier_matrices()
        _, sampled_factorized = sampled._patterned_fourier_matrices()
        for actual, expected in zip(constant_factorized, sampled_factorized):
            np.testing.assert_allclose(actual, expected, atol=2e-12)

    def test_operator_and_dense_backends_agree_for_factorized_layer(self):
        epsilon = np.ones((32, 3))
        epsilon[:13] = 4.0

        def solve(backend):
            crystal = Crystal((3, 1), lattice_pitch=1.0)
            crystal.add_layer_pixmap(
                "grating",
                epsilon,
                0.2,
                factorization="normal-vector",
                normal_vectors=(1.0, 0.0),
            )
            crystal.set_device(("grating",), fields_mask=False)
            crystal.set_source(1.7, te=0.0, tm=1.0, theta=7.0, phi=0.0)
            crystal.solve(backend=backend)
            return crystal.poynting_flux_end(only_total=True)

        dense = solve("dense")
        operator = solve("operator")
        np.testing.assert_allclose(operator, dense, atol=2e-11)


if __name__ == "__main__":
    unittest.main()
