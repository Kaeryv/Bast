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
)
run = ThreadRunner(4).run(case)
map_result = case.collect(run.evaluations)
transmission = map_result.series["T"]
```

`GuidedModeSpectrumCase` applies the same design to a spectrum: one wavelength
is one independent configuration, and `collect` locates the resonance after
all wavelengths return.

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
python scripts/validation_run.py lou-twist-map --threads 4 --output artifacts/validation
```
