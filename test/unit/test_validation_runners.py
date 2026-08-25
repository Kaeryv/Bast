from dataclasses import dataclass
from time import sleep
import unittest

from khepri.validation import (
    CaseConfiguration,
    CaseResult,
    SequentialRunner,
    ThreadRunner,
    ValidationCase,
)


@dataclass(frozen=True)
class OutOfOrderCase(ValidationCase):
    count: int = 6
    slug = "out-of-order"
    title = "Synthetic ordering check"
    description = "Tasks deliberately finish in reverse order."

    def configurations(self):
        for ordinal in range(self.count):
            yield CaseConfiguration.create(
                self.slug,
                str(ordinal),
                ordinal,
                (1, 1),
                {"value": ordinal},
            )

    def evaluate(self, configuration):
        value = int(configuration.parameter("value"))
        sleep((self.count - value) * 0.001)
        return CaseResult({"value": float(value * value)})

    def collect(self, evaluations):
        return CaseResult(
            {"sum": sum(item.result.observables["value"] for item in evaluations)}
        )


class RunnerTests(unittest.TestCase):
    def test_threaded_and_sequential_results_are_identical_and_ordered(self):
        case = OutOfOrderCase()
        serial = SequentialRunner().run(case)
        threaded = ThreadRunner(3).run(case)
        self.assertEqual(
            [item.configuration.key for item in serial.evaluations],
            [str(value) for value in range(case.count)],
        )
        self.assertEqual(
            [item.result.observables for item in serial.evaluations],
            [item.result.observables for item in threaded.evaluations],
        )
        self.assertEqual(case.collect(threaded.evaluations).observables["sum"], 55.0)

    def test_runner_rejects_duplicate_keys(self):
        case = OutOfOrderCase()
        values = list(case.configurations())
        duplicate = [values[0], values[0]]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            SequentialRunner().run(case, duplicate)

    def test_runner_accepts_a_stable_shard_with_global_ordinals(self):
        case = OutOfOrderCase()
        shard = tuple(case.configurations())[2:5]
        run = ThreadRunner(2).run(case, shard)
        self.assertEqual(
            [item.configuration.ordinal for item in run.evaluations],
            [2, 3, 4],
        )

    def test_runner_rejects_configuration_from_another_case(self):
        case = OutOfOrderCase()
        foreign = CaseConfiguration.create("other", "0", 0, 1, {})
        with self.assertRaisesRegex(ValueError, "belongs"):
            case.worker(foreign)

    def test_thread_count_is_validated(self):
        with self.assertRaises(ValueError):
            ThreadRunner(0)

    def test_empty_execution_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no configurations"):
            SequentialRunner().run(OutOfOrderCase(), ())


if __name__ == "__main__":
    unittest.main()
