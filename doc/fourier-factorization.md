# Fourier factorization

Khepri keeps its original classical Fourier product as the default.  Patterned
layers can opt into the normal-vector formulation of Popov–Nevière and Schuster
et al. when discontinuous permittivity makes the classical product converge
slowly, especially for TM fields and metals.

```python
crystal.add_layer_analytical(
    "lamellar",
    islands,
    epsilon_host,
    depth,
    factorization="normal-vector",
    normal_vectors=(1.0, 0.0),
)
```

The constant pair is the unit-cell normal direction for a one-dimensional
lamellar grating.  For a two-dimensional pixmap or analytical pattern, provide
a sampled, periodic continuation of the real normal-vector field:

```python
crystal.add_layer_pixmap(
    "pattern",
    epsilon_xy,
    depth,
    factorization="normal-vector",
    normal_vectors=(nx, ny),
)
```

`nx` and `ny` must have the same two-dimensional sampling grid, be finite and
nonzero as a pair at every sample. Khepri normalizes them pointwise. Their grid
must contain at least `(2*pw_x-1, 2*pw_y-1)` samples so all requested Fourier
differences exist. The normal field is geometric: it must not contain complex
values and should be continued smoothly and periodically away from material
boundaries. Khepri deliberately does not guess this continuation from a pixel
map because that choice is shape dependent and affects convergence.

The implementation forms the convolution matrices `[epsilon]` and
`[1/epsilon]`. Along the supplied local normal it applies Li's inverse rule,
`[1/epsilon]^-1`; along the tangent it retains Laurent's product. Constant
`(1, 0)` therefore reduces exactly to the established lamellar inverse-rule
formulation.

Both `backend="dense"` and `backend="operator"` accept factorized layers. The
operator backend changes stack composition and memory retention, not the
per-layer Maxwell eigenproblem, so both backends produce the same far-field
result.

References:

- P. Lalanne and G. M. Morris, “Highly improved convergence of the coupled-wave
  method for TM polarization,” JOSA A 13, 779–784 (1996),
  [doi:10.1364/JOSAA.13.000779](https://doi.org/10.1364/JOSAA.13.000779).
- E. Popov and M. Nevière, “Maxwell equations in Fourier space,” JOSA A 18,
  2886–2894 (2001),
  [doi:10.1364/JOSAA.18.002886](https://doi.org/10.1364/JOSAA.18.002886).
- T. Schuster et al., “Normal vector method for convergence improvement using
  the RCWA for crossed gratings,” JOSA A 24, 2880–2890 (2007),
  [doi:10.1364/JOSAA.24.002880](https://doi.org/10.1364/JOSAA.24.002880).
