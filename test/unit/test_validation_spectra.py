import unittest

from khepri.validation import (
    CaseResult,
    DisplacementSensitiveSlabCase,
    GuidedModeSpectrumCase,
)


class SpectrumContractTests(unittest.TestCase):
    def test_guided_mode_configs_are_one_wavelength_per_task(self):
        case = GuidedModeSpectrumCase(pw=(1, 1), samples=5, span_nm=2.0)
        configurations = tuple(case.configurations())
        self.assertEqual(len(configurations), 5)
        self.assertEqual(
            [item.ordinal for item in configurations],
            [0, 1, 2, 3, 4],
        )
        self.assertTrue(
            all("wavelength_nm" in item.parameters for item in configurations)
        )

    def test_spectrum_reference_is_explicitly_aggregate_and_graphical(self):
        case = GuidedModeSpectrumCase(pw=(1, 1), samples=5)
        aggregate = CaseResult({"resonance_nm": 618.0})
        reference = case.aggregate_reference(aggregate)
        self.assertEqual(reference.observables["resonance_nm"], 600.1)
        self.assertEqual(reference.kind, "graphical-literature-scalar")
        self.assertIn("18 nm", reference.uncertainty)
        self.assertAlmostEqual(case.aggregate_error(aggregate, reference), 17.9)

    def test_fan_configs_are_gap_shift_frequency_product(self):
        case = DisplacementSensitiveSlabCase(
            pw=(1, 1),
            frequency_range=(0.51, 0.53),
            samples=3,
            gaps=(1.35, 0.55),
            lateral_shifts=(0.0, 0.05),
        )
        configurations = tuple(case.configurations())
        self.assertEqual(len(configurations), 12)
        self.assertEqual(
            [item.ordinal for item in configurations], list(range(12))
        )
        self.assertEqual(
            [item.parameter("gap_over_a") for item in configurations[:6]],
            [1.35] * 6,
        )
        self.assertEqual(
            [item.parameter("shift_x_over_a") for item in configurations[:6]],
            [0.0] * 3 + [0.05] * 3,
        )


if __name__ == "__main__":
    unittest.main()
