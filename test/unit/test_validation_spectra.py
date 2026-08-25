import unittest

from khepri.validation import CaseResult, GuidedModeSpectrumCase


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


if __name__ == "__main__":
    unittest.main()
