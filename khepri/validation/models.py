"""Stable data contracts for reusable scientific validation cases."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from time import perf_counter
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np


PlaneWaves = Tuple[int, int]


def normalize_pw(value: Any) -> PlaneWaves:
    """Return a validated two-axis odd plane-wave truncation."""

    if isinstance(value, (int, np.integer)):
        value = (int(value), int(value))
    value = tuple(int(part) for part in value)
    if len(value) != 2 or any(part < 1 or part % 2 == 0 for part in value):
        raise ValueError("plane-wave grids must contain two positive odd integers")
    return value


def _freeze_parameter(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return tuple(_freeze_parameter(item) for item in value.tolist())
    if isinstance(value, Mapping):
        return tuple(
            (str(key), _freeze_parameter(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_parameter(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return value
    raise TypeError(
        "configuration parameters must contain scalar, tuple, list, array, or mapping values"
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, complex):
        return {"__complex__": [value.real, value.imag]}
    if isinstance(value, tuple):
        if value and all(
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
            for item in value
        ):
            return {key: _json_value(item) for key, item in value}
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _from_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        if set(value) == {"__complex__"}:
            real, imaginary = value["__complex__"]
            return complex(real, imaginary)
        return {
            str(name): _from_json_value(item) for name, item in value.items()
        }
    if isinstance(value, list):
        return tuple(_from_json_value(item) for item in value)
    return value


@dataclass(frozen=True)
class LiteratureReference:
    """Primary provenance for a case or external fixture."""

    citation: str
    url: str
    note: str = ""


@dataclass(frozen=True)
class CaseConfiguration:
    """One independent, serializable unit of scientific work.

    Parameters are stored as a sorted immutable tuple so configurations have a
    stable identity, are hashable, and can safely cross thread/process/Slurm
    boundaries.  ``parameters`` is exposed as a read-only mapping.
    """

    case_slug: str
    key: str
    ordinal: int
    pw: PlaneWaves
    _parameters: Tuple[Tuple[str, Any], ...] = field(repr=False)

    @classmethod
    def create(
        cls,
        case_slug: str,
        key: str,
        ordinal: int,
        pw: Any,
        parameters: Mapping[str, Any],
    ) -> "CaseConfiguration":
        frozen = tuple(
            (str(name), _freeze_parameter(value))
            for name, value in sorted(parameters.items())
        )
        return cls(str(case_slug), str(key), int(ordinal), normalize_pw(pw), frozen)

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "CaseConfiguration":
        """Reconstruct a task emitted by :meth:`to_dict`."""

        return cls.create(
            document["case_slug"],
            document["key"],
            document["ordinal"],
            document["pw"],
            {
                str(name): _from_json_value(value)
                for name, value in document["parameters"].items()
            },
        )

    @property
    def parameters(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self._parameters))

    def parameter(self, name: str) -> Any:
        try:
            return dict(self._parameters)[name]
        except KeyError as error:
            raise KeyError(f"configuration {self.key!r} has no parameter {name!r}") from error

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "case_slug": self.case_slug,
            "key": self.key,
            "ordinal": self.ordinal,
            "pw": list(self.pw),
            "parameters": {
                name: _json_value(value) for name, value in self._parameters
            },
        }


@dataclass
class CaseResult:
    """Observables produced by one independently executable configuration."""

    observables: Mapping[str, float]
    series: Mapping[str, np.ndarray] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    retained_bytes: int = 0

    def to_dict(self) -> dict:
        return {
            "observables": {
                str(name): _json_value(value)
                for name, value in self.observables.items()
            },
            "series": {
                str(name): _json_value(np.asarray(value))
                for name, value in self.series.items()
            },
            "metadata": {
                str(name): _json_value(value)
                for name, value in self.metadata.items()
            },
            "retained_bytes": int(self.retained_bytes),
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "CaseResult":
        return cls(
            {
                str(name): _from_json_value(value)
                for name, value in document["observables"].items()
            },
            series={
                str(name): np.asarray(_from_json_value(value))
                for name, value in document.get("series", {}).items()
            },
            metadata={
                str(name): _from_json_value(value)
                for name, value in document.get("metadata", {}).items()
            },
            retained_bytes=int(document.get("retained_bytes", 0)),
        )


@dataclass(frozen=True)
class ReferenceResult:
    """Trusted expected values and their provenance for one configuration."""

    observables: Mapping[str, float]
    kind: str
    source: Optional[LiteratureReference] = None
    fixture: Optional[str] = None
    uncertainty: str = ""


@dataclass(frozen=True)
class CaseEvaluation:
    """The result and task-local wall time for one configuration."""

    configuration: CaseConfiguration
    result: CaseResult
    wall_seconds: float

    def to_dict(self) -> dict:
        return {
            "configuration": self.configuration.to_dict(),
            "result": self.result.to_dict(),
            "wall_seconds": self.wall_seconds,
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "CaseEvaluation":
        return cls(
            CaseConfiguration.from_dict(document["configuration"]),
            CaseResult.from_dict(document["result"]),
            float(document["wall_seconds"]),
        )


@dataclass(frozen=True)
class CaseRun:
    """Ordered evaluations returned by any execution runner."""

    case_slug: str
    evaluations: Tuple[CaseEvaluation, ...]
    wall_seconds: float
    runner: str
    workers: int

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "case_slug": self.case_slug,
            "runner": self.runner,
            "workers": self.workers,
            "wall_seconds": self.wall_seconds,
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> "CaseRun":
        return cls(
            str(document["case_slug"]),
            tuple(
                CaseEvaluation.from_dict(item)
                for item in document["evaluations"]
            ),
            float(document["wall_seconds"]),
            str(document["runner"]),
            int(document["workers"]),
        )


class ValidationCase(ABC):
    """A physical experiment, independent from its execution policy.

    A case must enumerate deterministic configurations and evaluate each one
    without mutating shared state.  ``worker`` is deliberately small and safe
    to pass directly to ``ThreadPoolExecutor.map``.  A future process or Slurm
    runner can use the same configuration records and worker contract.
    """

    slug: str
    title: str
    description: str
    tags: Sequence[str] = ()
    references: Sequence[LiteratureReference] = ()

    @abstractmethod
    def configurations(self) -> Iterable[CaseConfiguration]:
        """Yield every independent task in stable, reproducible order."""

    @abstractmethod
    def evaluate(self, configuration: CaseConfiguration) -> CaseResult:
        """Evaluate one configuration without mutating the case."""

    def validate_configuration(self, configuration: CaseConfiguration) -> None:
        if configuration.case_slug != self.slug:
            raise ValueError(
                f"configuration belongs to {configuration.case_slug!r}, not {self.slug!r}"
            )

    def worker(self, configuration: CaseConfiguration) -> CaseEvaluation:
        """Thread/process-friendly worker callable for one configuration."""

        self.validate_configuration(configuration)
        started = perf_counter()
        result = self.evaluate(configuration)
        return CaseEvaluation(configuration, result, perf_counter() - started)

    def reference(
        self, configuration: CaseConfiguration
    ) -> Optional[ReferenceResult]:
        return None

    def error(
        self, evaluation: CaseEvaluation, reference: ReferenceResult
    ) -> float:
        keys = sorted(
            set(evaluation.result.observables).intersection(reference.observables)
        )
        if not keys:
            raise ValueError(
                f"{self.slug} has no observables shared with its reference"
            )
        return float(
            max(
                abs(
                    evaluation.result.observables[key]
                    - reference.observables[key]
                )
                for key in keys
            )
        )

    def collect(self, evaluations: Sequence[CaseEvaluation]) -> CaseResult:
        """Aggregate a case run; specialized cases may return maps or spectra."""

        if len(evaluations) != 1:
            raise NotImplementedError(
                f"{self.slug} must implement collect() for multiple configurations"
            )
        return evaluations[0].result

    def aggregate_reference(
        self, aggregate: CaseResult
    ) -> Optional[ReferenceResult]:
        """Return a reference for an observable defined only after collection."""

        return None

    def aggregate_error(
        self, aggregate: CaseResult, reference: ReferenceResult
    ) -> float:
        keys = sorted(set(aggregate.observables).intersection(reference.observables))
        if not keys:
            raise ValueError(
                f"{self.slug} aggregate has no observables shared with its reference"
            )
        return float(
            max(
                abs(aggregate.observables[key] - reference.observables[key])
                for key in keys
            )
        )
