"""Import checks executed against built wheel and source distributions."""

from importlib.metadata import version
from pathlib import Path

import khepri
import khepri.tmat.scattering
import khepri.validation
from khepri.validation.fixtures import load_manifest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
INSTALLED_ROOT = Path(khepri.__file__).resolve()

if SOURCE_ROOT in INSTALLED_ROOT.parents:
    raise RuntimeError(f"smoke test imported the source checkout: {INSTALLED_ROOT}")

manifest = load_manifest("lou_2021/manifest.json")
if manifest["case"] != "lou-twist-map":
    raise RuntimeError("bundled validation fixture is missing or malformed")

print(f"khepri {version('khepri')} imported from {INSTALLED_ROOT}")
