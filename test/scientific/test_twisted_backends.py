import os
import unittest
from math import radians

from numpy.testing import assert_allclose

from khepri import OperatorCapabilityError
from khepri.crystal import Crystal
from khepri.draw import Drawing
from khepri.expansion import Expansion
from khepri.layer import Layer


RUN_EXTENDED = os.environ.get("KHEPRI_RUN_EXTENDED") == "1"


def build_twisted(pw=(1, 1), *, fields_mask=False):
    pattern = Drawing((8, 8), 4.0)
    pattern.disc((0.0, 0.0), 0.25, 1.0)
    first = Expansion(pw)
    second = Expansion(pw)
    second.rotate(radians(13.0))
    extended = first + second
    crystal = Crystal.from_expansion(extended)
    crystal.add_layer(
        "upper",
        Layer.analytical(first, pattern.islands(), 4.0, 0.2),
        extended=True,
    )
    crystal.add_layer("gap", Layer.uniform(extended, 1.0, 0.3))
    crystal.add_layer(
        "lower",
        Layer.analytical(second, pattern.islands(), 4.0, 0.2),
        extended=True,
    )
    crystal.set_device(
        ["upper", "gap", "lower"], fields_mask=fields_mask
    )
    crystal.set_source(1.0 / 0.78, te=1.0, tm=1.0j)
    return crystal


class OperatorBackendTests(unittest.TestCase):
    def test_twisted_operator_matches_dense_without_hidden_materialization(self):
        operator = build_twisted()
        operator.solve(backend="operator")
        operator_rt = operator.poynting_flux_end()

        dense = build_twisted()
        dense.solve(backend="dense")
        assert_allclose(operator_rt, dense.poynting_flux_end(), atol=1e-12)
        self.assertEqual(operator.solve_diagnostics["backend"], "operator")
        self.assertFalse(operator.solve_diagnostics["materialized_dense"])
        with self.assertRaisesRegex(
            OperatorCapabilityError, "complete scattering matrix"
        ):
            _ = operator.Stot
        with self.assertRaisesRegex(OperatorCapabilityError, "backend='dense'"):
            _ = operator.S

    def test_backend_selection_preserves_default_and_supports_auto(self):
        historical = build_twisted()
        historical.solve()
        self.assertEqual(historical.solve_diagnostics["backend"], "dense")

        far_field = build_twisted()
        far_field.solve(backend="auto")
        self.assertEqual(far_field.solve_diagnostics["backend"], "operator")

        with_fields = build_twisted(fields_mask=[True, False, False])
        with_fields.solve(backend="auto")
        self.assertEqual(with_fields.solve_diagnostics["backend"], "dense")
        with self.assertRaisesRegex(
            OperatorCapabilityError, r"internal fields.*upper.*backend='dense'"
        ):
            with_fields.solve(backend="operator")

    def test_operator_field_request_explains_how_to_recover_fields(self):
        crystal = build_twisted()
        crystal.solve(backend="operator")
        with self.assertRaisesRegex(
            OperatorCapabilityError,
            r"field samples.*fields_mask.*backend='dense'",
        ):
            crystal.fields_coords_xy([[0.0]], [[0.0]], 0.1)

    def test_operator_ports_match_dense_for_asymmetric_uniform_stack(self):
        def build():
            crystal = Crystal(
                (3, 3), lattice_pitch=0.37, epsi=1.44, epse=2.25
            )
            crystal.add_layer_uniform("film", epsilon=3.7, depth=0.19)
            crystal.set_device(["film"], fields_mask=False)
            crystal.set_source(
                wavelength=1.13, te=0.4, tm=0.9, theta=17.0, phi=23.0
            )
            return crystal

        dense = build()
        dense.solve(backend="dense")
        operator = build()
        operator.solve(backend="operator")
        assert_allclose(
            operator.poynting_flux_end(), dense.poynting_flux_end(), atol=2e-12
        )
        operator_orders = operator.poynting_flux_end(only_total=False)
        dense_orders = dense.poynting_flux_end(only_total=False)
        for operator_flux, dense_flux in zip(operator_orders, dense_orders):
            assert_allclose(operator_flux[0], dense_flux[0], atol=2e-12)
            assert_allclose(operator_flux[1], dense_flux[1], atol=2e-12)

    def test_operator_supports_ordinary_patterned_layers(self):
        def build():
            expansion = Expansion((3, 3))
            pattern = Drawing((8, 8), 3.4)
            pattern.disc((0.0, 0.0), 0.22, 1.0)
            crystal = Crystal.from_expansion(expansion)
            crystal.add_layer(
                "pattern",
                Layer.analytical(
                    expansion, pattern.islands(), pattern.background, 0.18
                ),
            )
            crystal.add_layer_uniform("spacer", 1.7, 0.11)
            crystal.set_device(["pattern", "spacer"], fields_mask=False)
            crystal.set_source(1.31, te=0.7, tm=0.2)
            return crystal

        dense = build()
        dense.solve(backend="dense")
        operator = build()
        operator.solve(backend="operator")
        assert_allclose(
            operator.poynting_flux_end(), dense.poynting_flux_end(), atol=2e-11
        )

    @unittest.skipUnless(
        RUN_EXTENDED, "set KHEPRI_RUN_EXTENDED=1 for pw=3 twisted comparison"
    )
    def test_pw3_twisted_operator_matches_dense_flux(self):
        operator = build_twisted((3, 3))
        operator.solve(backend="operator")
        dense = build_twisted((3, 3))
        dense.solve(backend="dense")
        assert_allclose(
            operator.poynting_flux_end(), dense.poynting_flux_end(), atol=1e-9
        )


if __name__ == "__main__":
    unittest.main()
