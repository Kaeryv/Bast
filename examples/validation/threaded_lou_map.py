"""Compute a small Lou twist map with the reusable threaded case API."""

import numpy as np

from khepri.validation import LouTwistMapCase, ThreadRunner


def main():
    case = LouTwistMapCase(
        frequencies=tuple(np.linspace(0.74, 0.80, 4)),
        twists_degrees=tuple(np.linspace(5.0, 25.0, 5)),
        # pw=3 is the smallest truncation that resolves twist dependence.
        pw=(3, 3),
        polarization="rcp",
    )
    run = ThreadRunner(workers=4).run(case)
    result = case.collect(run.evaluations)
    print(result.series["T"])
    print(f"maximum energy defect: {result.observables['max_energy_defect']:.3e}")


if __name__ == "__main__":
    main()
