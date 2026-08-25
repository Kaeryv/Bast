# Reusable scientific validation cases

Khepri validation cases are executable physical experiments. A case is not a
test, example, benchmark, or execution scheduler. Those are independent
consumers of the same case:

```text
case: physical model + configurations + observables + references
  |
  +-- scientific test: pass/fail policy and tolerance
  +-- example/notebook: explanation and inspection
  +-- SequentialRunner / ThreadRunner: execution policy
  +-- JSON + Markdown: evidence and presentation
  +-- future ProcessRunner / Slurm runner: distributed execution
```

## The parallel contract

Every `ValidationCase` implements two primary methods:

```python
def configurations(self):
    """Yield deterministic, independent CaseConfiguration objects."""

def evaluate(self, configuration):
    """Construct fresh solver state and return one CaseResult."""
```

The inherited `worker(configuration)` method validates ownership, measures
task-local wall time, calls `evaluate`, and returns a `CaseEvaluation`. It is
safe to pass directly to a thread executor because case implementations are
immutable and every evaluation constructs its own mutable `Crystal`, `Layer`,
and `Expansion` objects.

Cases do not create threads. This keeps numerical code independent from
execution policy and prevents nested pools. The supplied runners are:

```python
run = SequentialRunner().run(case)
run = ThreadRunner(workers=8).run(case)
```

Both return evaluations in configuration order. `ThreadPoolExecutor.map`
preserves that order even when individual tasks finish out of order, so map
axes and spectra are deterministic.

Native BLAS/OpenMP thread counts should generally be one when case-level
threading is active. Task wall times are meaningful in a threaded run; a
thread cannot own a process-wide RSS or `tracemalloc` peak, so process memory
belongs to a future runner-level benchmark, not `CaseEvaluation`.

## Configuration records

`CaseConfiguration` contains:

- a stable case slug;
- a unique human-readable key;
- a contiguous ordinal;
- the plane-wave truncation;
- immutable named parameters.

It is hashable and JSON round-trippable:

```python
configuration = next(iter(case.configurations()))
document = configuration.to_dict()
same = CaseConfiguration.from_dict(document)
```

This is also the contract intended for later process and Slurm runners. A
distributed task receives a serialized case description/configuration, calls
the same worker, and returns a serialized `CaseEvaluation`.

Runners also accept a stable subset of configurations whose ordinals retain
their global values. This is the intended sharding mechanism: array task 17
can receive only its slice without renumbering it, and a merge step restores
global order before calling `collect`.

## Example: a Cartesian-product map

`LouTwistMapCase.configurations()` is equivalent to:

```python
for frequency, twist in product(frequencies, twists):
    yield CaseConfiguration(...)
```

Each pixel constructs an independent twisted stack and returns R/T/A. The
frequency-major sequence is collected into arrays with shape
`(n_frequency, n_twist)`:

```python
case = LouTwistMapCase(
    frequencies=(0.74, 0.76, 0.78),
    twists_degrees=(5.0, 10.0, 15.0),
    pw=(3, 3),
    backend="operator",
)
run = ThreadRunner(4).run(case)
map_result = case.collect(run.evaluations)
transmission = map_result.series["T"]
```

`GuidedModeSpectrumCase` applies the same design to a spectrum: one wavelength
is one independent configuration, and `collect` locates the resonance after
all wavelengths return.

`DisplacementSensitiveSlabCase` is a three-axis product:

```python
for gap, lateral_shift, frequency in product(gaps, shifts, frequencies):
    yield CaseConfiguration(...)
```

This matters because a complete Suh–Fan spectrum can be distributed without
sharing a mutable crystal between workers. Collection restores arrays with
shape `(n_gap, n_shift, n_frequency)` without pretending that the largest
sampled value is necessarily the narrow resonance. The default constructor
reproduces the published longitudinal gap list. The lateral-displacement
geometry is explicit:

```python
case = DisplacementSensitiveSlabCase.published_lateral_study(
    pw=(5, 5), samples=201
)
```

## Curated case catalogue and references

| Slug | Physical gate | Reference |
|---|---|---|
| `thin-film` | Fresnel interfaces, phase propagation, oblique TE/TM flux | Macleod, *Thin-Film Optical Filters*, DOI [10.1201/b21960](https://doi.org/10.1201/b21960) |
| `absorbing-film` | Complex epsilon, passive branch, oblique TE/TM R/T/A, zero-thickness limit | Born & Wolf, DOI [10.1017/CBO9781139644187](https://doi.org/10.1017/CBO9781139644187); Macleod, DOI [10.1201/b21960](https://doi.org/10.1201/b21960) |
| `absorbing-interface` | Complex-index Fresnel reflection and Poynting flux into a lossy half-space | Born & Wolf, DOI [10.1017/CBO9781139644187](https://doi.org/10.1017/CBO9781139644187) |
| `brewster` | TM Brewster zero and exterior flux normalization | Born & Wolf, *Principles of Optics*, DOI [10.1017/CBO9781139644187](https://doi.org/10.1017/CBO9781139644187) |
| `fabry-perot` | Repeated layers, Redheffer ordering, cavity phase, multiple reflections | Macleod, DOI [10.1201/b21960](https://doi.org/10.1201/b21960) |
| `lalanne-chrome-grating` | Patterned complex epsilon and TM Fourier-factorization convergence | Peng & Morris, DOI [10.1364/JOSAA.12.001087](https://doi.org/10.1364/JOSAA.12.001087); Lalanne & Morris, DOI [10.1364/JOSAA.13.000779](https://doi.org/10.1364/JOSAA.13.000779) |
| `luder-guided-mode` | Guided-mode spectral feature | Lüder et al. (2020), DOI [10.1007/s11082-020-02296-7](https://doi.org/10.1007/s11082-020-02296-7) |
| `lou-twist-map` | Extended-basis twisted bilayer map | Lou et al. (2021), DOI [10.1103/PhysRevLett.126.136101](https://doi.org/10.1103/PhysRevLett.126.136101) |
| `fan-displacement-sensitive-slabs` | Longitudinal and lateral displacement of coupled guided-resonant slabs | Suh et al. (2003), DOI [10.1063/1.1563739](https://doi.org/10.1063/1.1563739) |

The Fabry–Pérot case is a strong deterministic regression for stacking. Its
normal-incidence characteristic-matrix oracle is independent of Khepri's
scattering matrices and must agree pointwise. It is intentionally not sold as
a patterned-layer or Fourier-factorization benchmark: all layers are uniform.

The Suh–Fan case is more demanding scientifically. It uses the paper's square
lattice, air-hole radius `0.4a`, slab permittivity 12, thickness `0.55a`, and
published gap/shift sweeps. The bundled manifest records the plotted claims,
but contains no invented trace: the article does not provide numerical arrays.
Consequently the fast tests enforce execution order, thread safety, energy
conservation, and data shape. A literature reproduction must additionally
show plane-wave/frequency convergence before comparing resonance motion or
splitting against Figures 2 and 4.

## Absorbing-media conventions and open gates

Khepri assumes `exp(+i omega t)`. A passive material therefore has a negative
imaginary permittivity and refractive index, for example:

```python
index = 1.7 - 0.18j
epsilon = index**2
```

`AbsorbingFilmCase` is the hard, passing textbook gate. Its incident and exit
half-spaces are lossless, so `A = 1 - R - T` is unambiguously the power
absorbed in the finite film. It checks the full R/T/A triplet against an
independent oblique characteristic matrix, for both polarizations and several
thicknesses, including the zero-thickness limit.

Two cases deliberately expose unresolved scientific gaps rather than becoming
weak regressions:

- `absorbing-interface` currently fails the complex Fresnel reflectance and
  transmitted-flux checks because the exterior eigenspace assumes a lossless
  exit medium. Its scientific test is an expected failure until lossy
  half-spaces are implemented explicitly.
- `lalanne-chrome-grating` implements the classic normal-incidence chrome
  lamellar grating: period 0.25 µm, depth 0.20 µm, air-groove fraction 0.30,
  `n_Cr = 3.18 - 4.41j` at 0.55 µm, and a glass substrate of index 1.5. The
  literature modal reference is `T0 = 0.7028`. The current analytical
  patterned-layer formulation does not reach this value for TM polarization;
  this is the acceptance test for the planned normal-vector/Li Fourier
  factorization work. It is recorded as an expected failure, not redefined as
  self-convergence.

An unexpected success of either expected-failure test fails `unittest`, which
forces the marker and this documentation to be reviewed when the solver is
fixed.

## Adding a case

1. Add an immutable case class under `khepri.validation.cases` (the module can
   be split into a package as the catalogue grows).
2. Give it stable `slug`, `title`, `description`, `tags`, and primary sources.
3. Make `configurations()` deterministic and assign unique contiguous keys and
   ordinals.
4. Make `evaluate()` stateless: never cache a mutable Crystal or Layer on the
   case instance.
5. Return scalar comparison quantities in `CaseResult.observables`, arrays in
   `series`, and diagnostics/conventions in `metadata`.
6. Implement `reference(configuration)` when an independent reference exists.
7. Implement `collect(evaluations)` for spectra, maps, or multi-point studies.
   If the trusted observable exists only after collection, implement
   `aggregate_reference(aggregate)` rather than attaching it artificially to
   individual pixels.
8. Add the default case factory to `case_registry()`.
9. Add scientific tests with explicit acceptance tolerances. A case or fixture
   supplies evidence; the test owns pass/fail policy.

## Reference fixtures

External data lives under `khepri/validation/reference_data`. It is packaged
with Khepri and read through `fixture_path`/`load_manifest`.

Every literature fixture records its DOI, figure/table, geometry, convention,
data origin, uncertainty, and whether a numerical data file is actually
available. Khepri-generated JSON, maps, reports, or self-convergence results do
not belong in this directory and do not become reference truth.

Analytical references remain independent Python functions in
`khepri.validation.analytics`. Small formulas are easier to audit there than
as sampled fixture tables.

## Machine-readable output first

`write_json_run` writes configurations, individual results, references,
comparison errors, aggregate arrays, provenance, environment, runner, and wall
times. `write_markdown_run` derives a compact report from the completed run.
Figures should follow the same rule: generate them from stored JSON/NPZ data,
not by silently rerunning the physics.

Run a registered case with:

```bash
python scripts/validation_run.py thin-film --threads 1 --output artifacts/validation
python scripts/validation_run.py absorbing-film --threads 4 --output artifacts/validation
python scripts/validation_run.py absorbing-interface --threads 2 --output artifacts/validation
python scripts/validation_run.py fabry-perot --threads 4 --output artifacts/validation
python scripts/validation_run.py lalanne-chrome-grating --threads 4 --output artifacts/validation
python scripts/validation_run.py lou-twist-map --threads 4 --output artifacts/validation
```
