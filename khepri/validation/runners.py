"""Execution policies for validation cases."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Iterable, Optional, Tuple

from .models import CaseConfiguration, CaseRun, ValidationCase


def _ordered_configurations(
    case: ValidationCase,
    configurations: Optional[Iterable[CaseConfiguration]],
) -> Tuple[CaseConfiguration, ...]:
    values = tuple(case.configurations() if configurations is None else configurations)
    if not values:
        raise ValueError(f"{case.slug} produced no configurations")
    keys = [configuration.key for configuration in values]
    ordinals = [configuration.ordinal for configuration in values]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{case.slug} produced duplicate configuration keys")
    if ordinals != sorted(set(ordinals)):
        raise ValueError(
            f"{case.slug} configuration ordinals must be unique and increasing"
        )
    for configuration in values:
        case.validate_configuration(configuration)
    return values


@dataclass(frozen=True)
class SequentialRunner:
    """Reference execution policy with deterministic serial ordering."""

    def run(
        self,
        case: ValidationCase,
        configurations: Optional[Iterable[CaseConfiguration]] = None,
    ) -> CaseRun:
        values = _ordered_configurations(case, configurations)
        started = perf_counter()
        evaluations = tuple(case.worker(configuration) for configuration in values)
        return CaseRun(
            case.slug,
            evaluations,
            perf_counter() - started,
            type(self).__name__,
            1,
        )


@dataclass(frozen=True)
class ThreadRunner:
    """Run independent configurations in a notebook-safe thread pool.

    Every worker must construct its own mutable solver objects.  Native BLAS
    thread counts should normally be limited to one to avoid oversubscription.
    ``Executor.map`` preserves configuration order even when tasks finish out
    of order, so stored maps and spectra are deterministic.
    """

    workers: int

    def __post_init__(self):
        if self.workers < 1:
            raise ValueError("workers must be at least one")

    def run(
        self,
        case: ValidationCase,
        configurations: Optional[Iterable[CaseConfiguration]] = None,
    ) -> CaseRun:
        values = _ordered_configurations(case, configurations)
        started = perf_counter()
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            evaluations = tuple(executor.map(case.worker, values))
        return CaseRun(
            case.slug,
            evaluations,
            perf_counter() - started,
            type(self).__name__,
            self.workers,
        )
