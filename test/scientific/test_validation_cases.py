import unittest

import numpy as np

from khepri.validation import (
    AbsorbingFilmCase,
    AbsorbingInterfaceCase,
    BrewsterInterfaceCase,
    DisplacementSensitiveSlabCase,
    FabryPerotCavityCase,
    LalanneChromeGratingCase,
    LouTwistMapCase,
    SequentialRunner,
    ThinFilmCase,
    ThreadRunner,
)


class ScientificCaseTests(unittest.TestCase):
    def test_absorbing_film_matches_complex_characteristic_matrix(self):
        case = AbsorbingFilmCase(
            wavelengths=(0.83, 1.17),
            thicknesses=(0.0, 0.11, 0.43),
            angles_radians=(0.0, 0.47),
            polarizations=("s", "p"),
        )
        run = ThreadRunner(2).run(case)
        aggregate = case.collect(run.evaluations)
        for evaluation in run.evaluations:
            reference = case.reference(evaluation.configuration)
            self.assertLess(case.error(evaluation, reference), 3e-10)
        self.assertGreater(aggregate.observables["maximum_A"], 0.1)
        self.assertGreaterEqual(aggregate.observables["minimum_A"], -2e-12)
        self.assertLessEqual(aggregate.observables["maximum_R_plus_T"], 1 + 2e-12)
        np.testing.assert_allclose(
            aggregate.series["R"][:, 0, :, 0],
            aggregate.series["R"][:, 1, :, 0],
            atol=2e-14,
        )
        np.testing.assert_allclose(aggregate.series["A"][..., 0], 0.0, atol=2e-14)

    @unittest.expectedFailure
    def test_absorbing_halfspace_flux_matches_complex_fresnel(self):
        """Known gap: the exterior eigenspace assumes a lossless half-space."""

        case = AbsorbingInterfaceCase()
        run = ThreadRunner(2).run(case)
        for evaluation in run.evaluations:
            self.assertLess(case.error(evaluation, case.reference(evaluation.configuration)), 3e-10)

    def test_lalanne_chrome_case_records_published_geometry_and_absorption(self):
        case = LalanneChromeGratingCase(pw_values=(5, 9))
        configurations = tuple(case.configurations())
        self.assertEqual([item.pw for item in configurations], [(5, 1), (9, 1)])
        self.assertAlmostEqual(case.literature_transmission, 0.7028)
        self.assertAlmostEqual(
            case.secondary_literature_transmission, 0.69828465
        )
        self.assertAlmostEqual(case.chrome_ridge_fraction, 0.30)
        self.assertAlmostEqual(case.air_groove_fraction, 0.70)
        aggregate = case.collect(ThreadRunner(2).run(case).evaluations)
        self.assertGreaterEqual(aggregate.observables["minimum_A"], -2e-10)

    def test_lalanne_chrome_tm_transmission_reaches_published_value(self):
        case = LalanneChromeGratingCase(pw_values=(41,))
        aggregate = case.collect(SequentialRunner().run(case).evaluations)
        reference = case.aggregate_reference(aggregate)
        self.assertLess(case.aggregate_error(aggregate, reference), 0.015)
        self.assertLess(
            abs(
                aggregate.observables["highest_pw_T0"]
                - case.secondary_literature_transmission
            ),
            0.003,
        )

    def test_thin_film_matches_airy_for_s_and_p_in_parallel(self):
        case = ThinFilmCase(
            angles_radians=(0.0, 0.31),
            polarizations=("s", "p"),
        )
        run = ThreadRunner(2).run(case)
        self.assertEqual(len(run.evaluations), 4)
        for evaluation in run.evaluations:
            reference = case.reference(evaluation.configuration)
            self.assertLess(case.error(evaluation, reference), 3e-10)
            self.assertLess(abs(evaluation.result.observables["A"]), 2e-10)

    def test_brewster_zero(self):
        case = BrewsterInterfaceCase()
        evaluation = SequentialRunner().run(case).evaluations[0]
        self.assertLess(evaluation.result.observables["R"], 1e-12)
        self.assertLess(
            case.error(evaluation, case.reference(evaluation.configuration)),
            2e-11,
        )

    def test_fabry_perot_matches_independent_characteristic_matrix(self):
        case = FabryPerotCavityCase(
            wavelengths=(0.91, 1.0),
            gaps=(0.20, 0.47, 0.80),
            polarizations=("s", "p"),
        )
        run = ThreadRunner(2).run(case)
        aggregate = case.collect(run.evaluations)
        self.assertEqual(aggregate.series["T"].shape, (2, 2, 3))
        for evaluation in run.evaluations:
            reference = case.reference(evaluation.configuration)
            self.assertLess(case.error(evaluation, reference), 5e-10)
            self.assertLess(abs(evaluation.result.observables["A"]), 2e-10)
        np.testing.assert_allclose(
            aggregate.series["T"][0], aggregate.series["T"][1], atol=2e-12
        )

    def test_fan_case_is_parallel_safe_and_conserves_energy_at_pw1(self):
        case = DisplacementSensitiveSlabCase(
            pw=(1, 1),
            frequency_range=(0.52, 0.56),
            samples=3,
            gaps=(1.35, 0.55),
            lateral_shifts=(0.0, 0.05),
        )
        serial = SequentialRunner().run(case)
        threaded = ThreadRunner(2).run(case)
        serial_result = case.collect(serial.evaluations)
        threaded_result = case.collect(threaded.evaluations)
        np.testing.assert_allclose(
            serial_result.series["T"], threaded_result.series["T"], atol=2e-12
        )
        self.assertEqual(serial_result.series["T"].shape, (2, 2, 3))
        self.assertLess(serial_result.observables["max_energy_defect"], 1e-10)

    def test_fan_lateral_shift_changes_the_patterned_spectrum_at_pw3(self):
        case = DisplacementSensitiveSlabCase.published_lateral_study(
            pw=(3, 3),
            frequency_range=(0.52, 0.57),
            samples=11,
        )
        result = case.collect(ThreadRunner(2).run(case).evaluations)
        difference = np.max(np.abs(result.series["T"][0, 0] - result.series["T"][0, 1]))
        self.assertGreater(difference, 0.02)
        self.assertLess(result.observables["max_energy_defect"], 1e-10)

    def test_lou_map_is_cartesian_frequency_major_and_parallel_safe_at_pw1(self):
        case = LouTwistMapCase(
            frequencies=(0.74, 0.78),
            twists_degrees=(5.0, 15.0, 25.0),
            pw=(1, 1),
        )
        configurations = tuple(case.configurations())
        self.assertEqual(len(configurations), 6)
        self.assertEqual(
            [configuration.parameter("frequency_c_over_a") for configuration in configurations],
            [0.74, 0.74, 0.74, 0.78, 0.78, 0.78],
        )
        serial = SequentialRunner().run(case)
        threaded = ThreadRunner(3).run(case)
        serial_map = case.collect(serial.evaluations)
        threaded_map = case.collect(threaded.evaluations)
        np.testing.assert_allclose(serial_map.series["R"], threaded_map.series["R"], atol=2e-12)
        np.testing.assert_allclose(serial_map.series["T"], threaded_map.series["T"], atol=2e-12)
        self.assertEqual(serial_map.series["T"].shape, (2, 3))
        self.assertLess(serial_map.observables["max_energy_defect"], 1e-10)


if __name__ == "__main__":
    unittest.main()
