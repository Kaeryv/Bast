"""Read-only access to externally sourced validation fixtures."""

import json
from pathlib import Path
from typing import Mapping


FIXTURE_ROOT = Path(__file__).with_name("reference_data").resolve()


def fixture_path(name: str) -> Path:
    path = FIXTURE_ROOT.joinpath(*name.split("/")).resolve()
    try:
        path.relative_to(FIXTURE_ROOT)
    except ValueError as error:
        raise ValueError("fixture names must remain inside reference_data") from error
    if not path.is_file():
        raise FileNotFoundError(f"unknown validation fixture: {name}")
    return path


def load_manifest(name: str) -> Mapping[str, object]:
    """Load a JSON provenance manifest bundled with Khepri."""

    path = fixture_path(name)
    if path.suffix.lower() != ".json":
        raise ValueError("manifest fixtures must be JSON files")
    return json.loads(path.read_text(encoding="utf-8"))
