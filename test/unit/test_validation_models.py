import json
import unittest

import numpy as np

from khepri.validation import (
    CaseConfiguration,
    CaseEvaluation,
    CaseResult,
    CaseRun,
    normalize_pw,
)


class ValidationModelTests(unittest.TestCase):
    def test_configuration_is_stable_immutable_and_json_serializable(self):
        configuration = CaseConfiguration.create(
            "example",
            "pixel-7",
            7,
            3,
            {"twist": np.float64(12.5), "polarization": (1.0, 1.0j)},
        )
        self.assertEqual(configuration.pw, (3, 3))
        self.assertEqual(configuration.parameter("twist"), 12.5)
        with self.assertRaises(TypeError):
            configuration.parameters["twist"] = 15.0
        document = configuration.to_dict()
        json.dumps(document)
        self.assertEqual(CaseConfiguration.from_dict(document), configuration)

    def test_worker_and_run_records_round_trip_for_distributed_merging(self):
        configuration = CaseConfiguration.create("case", "point", 3, 1, {"x": 2.0})
        evaluation = CaseEvaluation(
            configuration,
            CaseResult({"T": 0.75}, series={"field": np.asarray([1.0, 2.0])}),
            0.125,
        )
        run = CaseRun("case", (evaluation,), 0.13, "ThreadRunner", 4)
        document = run.to_dict()
        json.dumps(document)
        restored = CaseRun.from_dict(document)
        self.assertEqual(restored.case_slug, run.case_slug)
        self.assertEqual(restored.evaluations[0].configuration, configuration)
        self.assertEqual(restored.evaluations[0].result.observables["T"], 0.75)
        np.testing.assert_allclose(
            restored.evaluations[0].result.series["field"],
            (1.0, 2.0),
        )
    def test_result_serialization_handles_arrays_and_complex_metadata(self):
        result = CaseResult(
            {"R": np.float64(0.25)},
            series={"axis": np.asarray([1.0, 2.0])},
            metadata={"polarization": 1.0j},
        )
        document = result.to_dict()
        self.assertEqual(document["series"]["axis"], [1.0, 2.0])
        self.assertEqual(document["metadata"]["polarization"]["__complex__"][1], 1.0)
        json.dumps(document)

    def test_plane_wave_validation(self):
        self.assertEqual(normalize_pw(3), (3, 3))
        self.assertEqual(normalize_pw((5, 1)), (5, 1))
        for invalid in (0, 2, (3,), (3, 2)):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    normalize_pw(invalid)


if __name__ == "__main__":
    unittest.main()
