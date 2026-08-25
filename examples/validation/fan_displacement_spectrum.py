"""Run a small parallel Suh–Fan longitudinal-displacement spectrum.

Primary reference: W. Suh et al., Applied Physics Letters 82, 1999 (2003),
DOI 10.1063/1.1563739.  PW3 is useful for exercising the workflow, not for a
quantitative literature claim; promote interesting features to PW5/PW7 and a
finer frequency grid.
"""

import argparse

from khepri.validation import DisplacementSensitiveSlabCase, ThreadRunner


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pw", type=int, default=3)
    parser.add_argument("--samples", type=int, default=41)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)

    case = DisplacementSensitiveSlabCase(
        pw=(args.pw, args.pw),
        samples=args.samples,
    )
    run = ThreadRunner(args.threads).run(case)
    result = case.collect(run.evaluations)
    frequencies = result.series["frequencies"]
    print("gap/a   coarse internal-max candidates (c/a, T)")
    for gap, transmission in zip(result.series["gaps"], result.series["T"][:, 0]):
        indices = [
            index
            for index in range(1, len(frequencies) - 1)
            if transmission[index] > transmission[index - 1]
            and transmission[index] > transmission[index + 1]
        ]
        candidates = ", ".join(
            f"({frequencies[index]:.5f}, {transmission[index]:.4f})"
            for index in indices
        )
        print(f"{gap:5.2f}   {candidates or 'none resolved'}")
    print(f"maximum energy defect: {result.observables['max_energy_defect']:.3e}")
    print("Candidates are not fitted resonances; refine frequency and PW before comparison.")


if __name__ == "__main__":
    main()
