import unittest

import numpy as np
from numpy.testing import assert_allclose

from khepri import Crystal
from khepri.draw import Drawing


class FieldRetentionScientificTests(unittest.TestCase):
    @staticmethod
    def solve(fields_mask):
        drawing = Drawing((32, 32), 4.0)
        drawing.disc((0.0, 0.0), 0.23, 1.0)
        crystal = Crystal((3, 3), epsi=1.0, epse=2.25)
        crystal.add_layer_analytical(
            "pattern", drawing.islands(), drawing.background, 0.17
        )
        crystal.add_layer_uniform("film", epsilon=2.0, depth=0.31)
        crystal.set_device(
            ["pattern", "film"], fields_mask=fields_mask
        )
        crystal.set_source(
            wavelength=1.07, te=0.8, tm=0.6j, theta=11.0, phi=17.0
        )
        crystal.solve()
        return crystal

    def test_rta_and_total_s_are_unchanged_when_stacks_are_dropped(self):
        total_only = self.solve([False, False])
        retained = self.solve([True, True])
        assert_allclose(total_only.Stot, retained.Stot, atol=2e-12)
        assert_allclose(
            total_only.poynting_flux_end(),
            retained.poynting_flux_end(),
            atol=2e-12,
        )
        reflection, transmission = total_only.poynting_flux_end()
        self.assertLess(abs(1.0 - reflection - transmission), 2e-11)
        self.assertEqual(total_only.stacking_matrices, [])
        self.assertEqual(total_only.stacking_reverse_matrices, [])


if __name__ == "__main__":
    unittest.main()
