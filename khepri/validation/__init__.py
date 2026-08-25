"""Reusable scientific experiments, references, runners, and reports."""

from .analytics import (
    brewster_angle,
    fresnel_interface,
    multilayer_stack,
    multilayer_stack_oblique,
    single_film,
)
from .cases import (
    AbsorbingFilmCase,
    AbsorbingInterfaceCase,
    BrewsterInterfaceCase,
    DisplacementSensitiveSlabCase,
    FabryPerotCavityCase,
    GuidedModeSpectrumCase,
    LalanneChromeGratingCase,
    LouTwistMapCase,
    ThinFilmCase,
    case_registry,
)
from .fixtures import FIXTURE_ROOT, fixture_path, load_manifest
from .models import (
    CaseConfiguration,
    CaseEvaluation,
    CaseResult,
    CaseRun,
    LiteratureReference,
    ReferenceResult,
    ValidationCase,
    normalize_pw,
)
from .reporting import write_json_run, write_markdown_run
from .runners import SequentialRunner, ThreadRunner


__all__ = [
    "AbsorbingFilmCase",
    "AbsorbingInterfaceCase",
    "BrewsterInterfaceCase",
    "CaseConfiguration",
    "CaseEvaluation",
    "CaseResult",
    "CaseRun",
    "DisplacementSensitiveSlabCase",
    "FIXTURE_ROOT",
    "FabryPerotCavityCase",
    "GuidedModeSpectrumCase",
    "LalanneChromeGratingCase",
    "LiteratureReference",
    "LouTwistMapCase",
    "ReferenceResult",
    "SequentialRunner",
    "ThinFilmCase",
    "ThreadRunner",
    "ValidationCase",
    "brewster_angle",
    "case_registry",
    "fixture_path",
    "fresnel_interface",
    "load_manifest",
    "multilayer_stack",
    "multilayer_stack_oblique",
    "normalize_pw",
    "single_film",
    "write_json_run",
    "write_markdown_run",
]
