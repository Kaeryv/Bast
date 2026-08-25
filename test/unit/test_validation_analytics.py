import unittest

from khepri.validation import (
    fresnel_interface,
    multilayer_stack,
    multilayer_stack_oblique,
    single_film,
)


class AnalyticalReferenceTests(unittest.TestCase):
    def test_multilayer_reduces_to_single_film_at_normal_incidence(self):
        expected = single_film(1.0, 4.0, 2.25, 0.17, 0.93, 0.0, "s")
        actual = multilayer_stack(1.0, 2.25, ((4.0, 0.17),), 0.93)
        for name in ("R", "T", "A"):
            self.assertAlmostEqual(actual[name], expected[name], places=13)

    def test_lossless_multilayer_conserves_energy(self):
        actual = multilayer_stack(
            1.0,
            1.0,
            ((4.0, 0.125), (1.0, 0.37), (4.0, 0.125)),
            1.0,
        )
        self.assertLess(abs(actual["A"]), 2e-14)

    def test_passive_complex_film_has_positive_absorption(self):
        epsilon = (1.7 - 0.18j) ** 2
        for polarization in ("s", "p"):
            actual = multilayer_stack_oblique(
                1.0,
                2.25,
                ((epsilon, 0.23),),
                1.1,
                0.47,
                polarization,
            )
            self.assertGreater(actual["A"], 0.1)
            self.assertLess(actual["R"] + actual["T"], 1.0)
            self.assertAlmostEqual(sum(actual.values()), 1.0, places=14)

    def test_zero_thickness_complex_film_reduces_to_fresnel_interface(self):
        epsilon = (1.7 - 0.18j) ** 2
        for polarization in ("s", "p"):
            actual = multilayer_stack_oblique(
                1.0,
                2.25,
                ((epsilon, 0.0),),
                0.91,
                0.31,
                polarization,
            )
            expected = fresnel_interface(1.0, 2.25, 0.31, polarization)
            for name in ("R", "T", "A"):
                self.assertAlmostEqual(actual[name], expected[name], places=13)


if __name__ == "__main__":
    unittest.main()
