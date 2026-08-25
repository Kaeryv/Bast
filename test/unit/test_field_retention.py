import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from khepri import Crystal, FieldsNotRetained, FieldsNotRetainedWarning
from khepri.layer import stack_layers


class FieldRetentionTests(unittest.TestCase):
    @staticmethod
    def build(fields_mask=False):
        crystal = Crystal((1, 1))
        crystal.add_layer_uniform("film", epsilon=4.0, depth=0.2)
        crystal.set_device(["film"], fields_mask=fields_mask)
        crystal.set_source(1.1, te=1.0, tm=0.0)
        return crystal

    def test_all_false_mask_skips_partial_stack_builder(self):
        crystal = self.build(fields_mask=[False])
        with patch("khepri.crystal.stack_layers") as partial_builder:
            crystal.solve()
        partial_builder.assert_not_called()
        self.assertFalse(crystal.internal_fields_requested)
        self.assertEqual(crystal.device_fields_mask, (False,))
        self.assertEqual(crystal.stacking_matrices, [])
        self.assertEqual(crystal.stacking_reverse_matrices, [])

    def test_total_only_stack_matches_legacy_retained_total(self):
        crystal = self.build(fields_mask=False)
        crystal.solve()
        layers = [crystal.layers[name] for name in crystal.global_stacking]
        forwards, reverses, retained_total = stack_layers(
            crystal.expansion.pw, layers, [True, False, True]
        )
        assert_allclose(crystal.Stot, retained_total, atol=1e-14)
        retained = [
            matrix
            for matrix in (*forwards, *reverses)
            if matrix is not None
        ]
        self.assertEqual(len(retained), 4)
        self.assertEqual(
            sum(matrix.nbytes for matrix in retained),
            4 * crystal.Stot.nbytes,
        )

    def test_disabled_layer_returns_specific_value_and_warning(self):
        crystal = self.build(fields_mask=False)
        crystal.solve()
        coordinates = np.asarray([[0.0]])
        with self.assertWarnsRegex(
            FieldsNotRetainedWarning, "asked fields in 'film' layer"
        ):
            result = crystal.fields_coords_xy(
                coordinates, coordinates, z=0.1
            )
        self.assertIsInstance(result, FieldsNotRetained)
        self.assertEqual(result.layer_name, "film")
        self.assertEqual(result.layer_index, 1)
        self.assertEqual(result.z, 0.1)
        self.assertEqual(crystal.stacking_matrices, [])
        self.assertEqual(crystal.stacking_reverse_matrices, [])

    def test_exterior_fields_remain_available_without_partial_stacks(self):
        crystal = self.build(fields_mask=False)
        crystal.solve()
        legacy = self.build(fields_mask=False)
        legacy.internal_fields_requested = True
        legacy.solve()
        coordinates = np.asarray([[0.0]])
        for z in (-0.1, 0.3):
            result = crystal.fields_coords_xy(coordinates, coordinates, z=z)
            self.assertNotIsInstance(result, FieldsNotRetained)
            electric, magnetic = result
            self.assertEqual(electric.shape, (3, 1, 1))
            self.assertEqual(magnetic.shape, (3, 1, 1))
            legacy_fields = legacy.fields_coords_xy(
                coordinates, coordinates, z=z
            )
            assert_allclose(result[0], legacy_fields[0], atol=2e-14)
            assert_allclose(result[1], legacy_fields[1], atol=2e-14)
        self.assertEqual(crystal.stacking_matrices, [])
        self.assertEqual(crystal.stacking_reverse_matrices, [])

    def test_one_true_marker_does_not_enable_another_occurrence(self):
        crystal = Crystal((1, 1))
        crystal.add_layer_uniform("film", epsilon=4.0, depth=0.2)
        crystal.set_device(
            ["film", "film"], fields_mask=[True, False]
        )
        crystal.set_source(1.1, te=1.0, tm=0.0)
        crystal.solve()
        coordinates = np.asarray([[0.0]])
        retained = crystal.fields_coords_xy(
            coordinates, coordinates, z=0.1
        )
        self.assertNotIsInstance(retained, FieldsNotRetained)
        with self.assertWarnsRegex(
            FieldsNotRetainedWarning, "asked fields in 'film' layer"
        ):
            disabled = crystal.fields_coords_xy(
                coordinates, coordinates, z=0.3
            )
        self.assertIsInstance(disabled, FieldsNotRetained)

    def test_volume_preflights_single_pass_depth_iterators(self):
        crystal = self.build(fields_mask=False)
        crystal.solve()
        coordinates = np.asarray([[0.0]])
        depths = (value for value in (-0.1, 0.1))
        with self.assertWarnsRegex(
            FieldsNotRetainedWarning, "asked fields in 'film' layer"
        ):
            result = crystal.fields_volume(coordinates, coordinates, depths)
        self.assertIsInstance(result, FieldsNotRetained)
        self.assertEqual(result.z, 0.1)

    def test_true_layer_keeps_internal_field_behavior(self):
        crystal = self.build(fields_mask=True)
        crystal.solve()
        self.assertTrue(crystal.internal_fields_requested)
        self.assertEqual(crystal.device_fields_mask, (True,))
        self.assertTrue(any(
            matrix is not None for matrix in crystal.stacking_matrices
        ))
        self.assertTrue(any(
            matrix is not None for matrix in crystal.stacking_reverse_matrices
        ))
        coordinates = np.asarray([[0.0]])
        electric, magnetic = crystal.fields_coords_xy(
            coordinates, coordinates, z=0.1
        )
        self.assertEqual(electric.shape, (3, 1, 1))
        self.assertEqual(magnetic.shape, (3, 1, 1))

    def test_mask_shape_values_and_numpy_boolean_are_supported(self):
        crystal = Crystal((1, 1))
        crystal.add_layer_uniform("film", epsilon=4.0, depth=0.2)
        with self.assertRaisesRegex(ValueError, "one boolean per device layer"):
            crystal.set_device(["film"], fields_mask=[])
        with self.assertRaisesRegex(TypeError, "values must be booleans"):
            crystal.set_device(["film"], fields_mask=[1])
        crystal.set_device(["film"], fields_mask=np.bool_(False))
        self.assertEqual(crystal.device_fields_mask, (False,))

    def test_reconfiguring_device_clears_stale_field_intent(self):
        crystal = self.build(fields_mask=True)
        self.assertTrue(crystal.layers["film"].fields)
        crystal.set_device(["film"], fields_mask=False)
        self.assertFalse(crystal.internal_fields_requested)
        self.assertFalse(crystal.layers["film"].fields)


if __name__ == "__main__":
    unittest.main()
