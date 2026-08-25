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

Current primary-source manifests:

- `lalanne_morris_1996`: DOI `10.1364/JOSAA.13.000779`, with geometry
  provenance cross-checked against DOI `10.1364/JOSAA.12.001087` and
  `10.1109/JLT.2009.2027343`;
- `luder_2020`: DOI `10.1007/s11082-020-02296-7`;
- `lou_2021`: DOI `10.1103/PhysRevLett.126.136101`;
- `suh_fan_2003`: DOI `10.1063/1.1563739`.
