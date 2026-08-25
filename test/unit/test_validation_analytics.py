import unittest

from khepri.validation import multilayer_stack, single_film


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


if __name__ == "__main__":
    unittest.main()
