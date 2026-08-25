# Operator RCWA backend

The operator backend is a source-specific R/T solver. It composes a stack by
solving its interface amplitudes, rather than constructing the complete dense
scattering matrix of every partial stack.

## User API

The historical solve remains unchanged:

```python
crystal.solve()                  # identical to backend="dense"
```

Opt into the memory-efficient path at the solve call:

```python
crystal.solve(backend="operator")
R, T = crystal.poynting_flux_end()
R_orders, T_orders = crystal.poynting_flux_end(only_total=False)
```

`backend="auto"` selects the operator backend for a compatible twisted stack
when no internal layer is marked for field retention. Otherwise it selects the
dense backend. `crystal.solve_diagnostics["backend"]` records the actual choice.

The same call accepts ordinary uniform layers, ordinary patterned layers, and
extended/twisted layers. Uniform layers are stored as independent 2x2
polarization blocks. An ordinary patterned layer retains its conventional
per-layer dense RCWA matrix, but the stack is still solved without dense
partial or total S matrices. An extended layer retains shifted base-layer
matrices and applies them directly in the joint reciprocal space.

## Deliberate capability boundary

An operator solve computes reflected and transmitted modal vectors for the one
source configured by `set_source`. It is not the complete multi-input S matrix,
and it does not retain the forward/reverse stacks or eigenspaces required for
internal fields.

These operations therefore raise `OperatorCapabilityError`:

- `crystal.S` or `crystal.Stot` after an operator solve;
- internal electric or magnetic field sampling;
- explicitly selecting the operator backend while any device
  `fields_mask` entry is true.

The exception names the unavailable quantity and explains how to recover it:

```python
crystal.solve(backend="dense")
```

This is intentional. Khepri never silently materializes a potentially
multi-gigabyte matrix after an operator solve.

## Iterative failures

The interface network is solved with preconditioned GMRES. Failure raises
`IterativeConvergenceError`, which reports the final residual, iteration count,
linear-system dimension, and estimated storage of the dense layer matrices.
It does not silently fall back to a dense solve, since such a fallback can
exceed available RAM. The caller can adjust `rtol`, `atol`, or `maxiter`, reduce
the truncation, or deliberately re-run with `backend="dense"`.

Useful diagnostics are stored in `crystal.solve_diagnostics`, including solve
time, iteration count, residual, sparse-system nonzeros, temporary operator
storage, retained result bytes, and an estimate of the equivalent dense layer
storage.

## Bounded implementation benchmark

The feature gate used the Lou twisted bilayer at `f a/c = 0.78`, 13 degrees,
RCP incidence. These are isolated-process measurements on the development
arm64 workstation; they document scale rather than promise portable timing.

| Base PW | Backend | Wall time | Peak RSS | Temporary operator storage | Dense-layer estimate |
|---:|---|---:|---:|---:|---:|
| 5x5 | dense | 7.16 s | 1.19 GiB | — | 0.47 GiB |
| 5x5 | operator | 0.70 s | 0.20 GiB | 8.10 MiB | 0.47 GiB |
| 7x7 | operator | 6.52 s | not recorded | 59.24 MiB | 6.87 GiB |

At PW5 the operator and dense totals agreed to `1.8e-14`; the operator path
was 10.3x faster and reduced measured peak RSS by 83.5%. At PW7, four GMRES
iterations reached a relative residual of `5.6e-11`. A PW7 dense run was not
repeated locally because the estimate itself is the capacity risk this backend
is designed to avoid.
