# Scientific reference data

This directory contains externally trusted fixtures and their provenance. It
does not contain Khepri-generated benchmark output.

Each literature fixture has a manifest recording:

- primary citation and DOI;
- exact figure, table, or supplementary source;
- geometry and convention information needed to interpret the data;
- whether values are tabulated, digitized, analytical, or independently
  generated;
- uncertainty and known ambiguities.

Generated JSON, Markdown, plots, and benchmark arrays belong under an output
directory chosen by the runner. They must never silently become reference
truth.
