import unittest

import numpy as np

from khepri.validation import (
    BrewsterInterfaceCase,
    LouTwistMapCase,
    SequentialRunner,
    ThinFilmCase,
    ThreadRunner,
)


class ScientificCaseTests(unittest.TestCase):
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
