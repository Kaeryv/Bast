"""Curated physical cases shared by tests, examples, and runners."""

from dataclasses import dataclass
from itertools import product
from math import radians
from typing import Iterable, Sequence, Tuple

import numpy as np

from khepri.crystal import Crystal
from khepri.draw import Drawing
from khepri.expansion import Expansion
from khepri.layer import Layer

from .analytics import brewster_angle, fresnel_interface, single_film
from .fixtures import load_manifest
from .models import (
    CaseConfiguration,
    CaseEvaluation,
    CaseResult,
    LiteratureReference,
    PlaneWaves,
    ReferenceResult,
    ValidationCase,
    normalize_pw,
)


def _rta(crystal: Crystal) -> Tuple[float, float, float]:
    reflection, transmission = crystal.poynting_flux_end()
    reflection = float(reflection)
    transmission = float(transmission)
    return reflection, transmission, 1.0 - reflection - transmission


@dataclass(frozen=True)
class ThinFilmCase(ValidationCase):
    """Uniform film with a closed-form Airy reference."""

    pw_values: Tuple[PlaneWaves, ...] = ((1, 1),)
    wavelengths: Tuple[float, ...] = (1.0,)
    angles_radians: Tuple[float, ...] = (0.0,)
    polarizations: Tuple[str, ...] = ("s",)
    epsilon_incident: float = 1.0
    epsilon_film: float = 4.0
    epsilon_substrate: float = 2.25
    thickness: float = 0.125

    slug = "thin-film"
    title = "Single dielectric thin film"
    description = "Uniform film checked against an independent Airy formula."
    tags = ("fast", "analytic", "integration", "thin-film")
    references = (
        LiteratureReference(
            "H. A. Macleod, Thin-Film Optical Filters, 4th ed.",
            "https://doi.org/10.1201/b21960",
            "Standard characteristic-matrix/Airy solution.",
        ),
    )

    def configurations(self) -> Iterable[CaseConfiguration]:
        ordinal = 0
        for pw, wavelength, theta, polarization in product(
            self.pw_values,
            self.wavelengths,
            self.angles_radians,
            self.polarizations,
        ):
            pw = normalize_pw(pw)
            key = (
                f"pw={pw[0]}x{pw[1]};wavelength={wavelength:.12g};"
                f"theta={theta:.12g};pol={polarization.lower()}"
            )
            yield CaseConfiguration.create(
                self.slug,
                key,
                ordinal,
                pw,
                {
                    "wavelength": wavelength,
                    "theta_radians": theta,
                    "polarization": polarization.lower(),
                },
            )
            ordinal += 1

    def evaluate(self, configuration: CaseConfiguration) -> CaseResult:
        wavelength = float(configuration.parameter("wavelength"))
        theta = float(configuration.parameter("theta_radians"))
        polarization = str(configuration.parameter("polarization"))
        crystal = Crystal(
            configuration.pw,
            lattice_pitch=0.37,
            epsi=self.epsilon_incident,
            epse=self.epsilon_substrate,
        )
        crystal.add_layer_uniform("film", self.epsilon_film, self.thickness)
        crystal.set_device(["film"], fields_mask=False)
        te, tm = ((1.0, 0.0) if polarization in ("s", "te") else (0.0, 1.0))
        crystal.set_source(
            wavelength,
            te=te,
            tm=tm,
            theta=np.degrees(theta),
            phi=0.0,
        )
        crystal.solve()
        reflection, transmission, absorption = _rta(crystal)
        return CaseResult(
            {"R": reflection, "T": transmission, "A": absorption},
            metadata={"reference_plane": "exterior flux"},
            retained_bytes=crystal.Stot.nbytes,
        )

    def reference(self, configuration: CaseConfiguration) -> ReferenceResult:
        values = single_film(
            self.epsilon_incident,
            self.epsilon_film,
            self.epsilon_substrate,
            self.thickness,
            float(configuration.parameter("wavelength")),
            float(configuration.parameter("theta_radians")),
            str(configuration.parameter("polarization")),
        )
        return ReferenceResult(values, "analytical", self.references[0])

    def collect(self, evaluations: Sequence[CaseEvaluation]) -> CaseResult:
        evaluations = tuple(
            sorted(evaluations, key=lambda item: item.configuration.ordinal)
        )
        return CaseResult(
            {},
            series={
                "R": np.asarray([item.result.observables["R"] for item in evaluations]),
                "T": np.asarray([item.result.observables["T"] for item in evaluations]),
                "A": np.asarray([item.result.observables["A"] for item in evaluations]),
            },
            metadata={"configuration_count": len(evaluations)},
        )


@dataclass(frozen=True)
class BrewsterInterfaceCase(ValidationCase):
    pw: PlaneWaves = (1, 1)
    wavelength: float = 1.0
    epsilon_incident: float = 1.0
    epsilon_transmitted: float = 2.25

    slug = "brewster"
    title = "Brewster-angle interface"
    description = "A lossless interface has zero TM reflection at Brewster incidence."
    tags = ("fast", "analytic", "integration", "brewster")

    @property
    def theta(self) -> float:
        return brewster_angle(self.epsilon_incident, self.epsilon_transmitted)

    def configurations(self) -> Iterable[CaseConfiguration]:
        yield CaseConfiguration.create(
            self.slug,
            "brewster-tm",
            0,
            self.pw,
            {
                "wavelength": self.wavelength,
                "theta_radians": self.theta,
                "polarization": "p",
            },
        )

    def evaluate(self, configuration: CaseConfiguration) -> CaseResult:
        crystal = Crystal(
            configuration.pw,
            lattice_pitch=0.37,
            epsi=self.epsilon_incident,
            epse=self.epsilon_transmitted,
        )
        crystal.set_device([])
        crystal.set_source(
            float(configuration.parameter("wavelength")),
            te=0.0,
            tm=1.0,
            theta=np.degrees(float(configuration.parameter("theta_radians"))),
            phi=0.0,
        )
        crystal.solve()
        reflection, transmission, absorption = _rta(crystal)
        return CaseResult(
            {"R": reflection, "T": transmission, "A": absorption},
            metadata={"theta_degrees": np.degrees(self.theta)},
            retained_bytes=crystal.Stot.nbytes,
        )

    def reference(self, configuration: CaseConfiguration) -> ReferenceResult:
        values = fresnel_interface(
            self.epsilon_incident,
            self.epsilon_transmitted,
            float(configuration.parameter("theta_radians")),
            "p",
        )
        return ReferenceResult(values, "analytical")


@dataclass(frozen=True)
class GuidedModeSpectrumCase(ValidationCase):
    """Lüder et al. spectrum split into independently executable wavelengths."""

    pw: PlaneWaves = (7, 1)
    polarization: str = "te"
    samples: int = 41
    span_nm: float = 20.0

    slug = "luder-guided-mode"
    title = "Lüder guided-mode resonance spectrum"
    description = "One wavelength per task, aggregated into a transmission spectrum."
    tags = ("slow", "literature", "spectrum", "parallel")
    references = (
        LiteratureReference(
            "H. Lüder, M. Paulsen, and M. Gerken, Opt. Quantum Electron. 52, 180 (2020)",
            "https://doi.org/10.1007/s11082-020-02296-7",
            "Figure 3 reports TE and TM resonance wavelengths.",
        ),
    )

    @property
    def manifest(self):
        return load_manifest("luder_2020/manifest.json")

    @property
    def expected_nm(self) -> float:
        return float(self.manifest["reference_resonance_nm"][self.polarization.lower()])

    def wavelengths_nm(self) -> np.ndarray:
        return np.linspace(
            self.expected_nm - self.span_nm,
            self.expected_nm + self.span_nm,
            self.samples,
        )

    def configurations(self) -> Iterable[CaseConfiguration]:
        for ordinal, wavelength_nm in enumerate(self.wavelengths_nm()):
            yield CaseConfiguration.create(
                self.slug,
                f"wavelength_nm={wavelength_nm:.12g}",
                ordinal,
                self.pw,
                {
                    "wavelength_nm": float(wavelength_nm),
                    "polarization": self.polarization.lower(),
                },
            )

    def evaluate(self, configuration: CaseConfiguration) -> CaseResult:
        period_nm = float(self.manifest["geometry"]["period_nm"])
        pattern = Drawing((8, 8), 1.0)
        pattern.rectangle((0.0, 0.0), (0.6, 1.0), 1.5**2)
        crystal = Crystal(configuration.pw, lattice_pitch=1.0, epsi=1.0, epse=1.5**2)
        crystal.add_layer_analytical(
            "grating", pattern.islands(), pattern.background, 60.0 / period_nm
        )
        crystal.add_layer_uniform("waveguide", 2.3**2, 85.0 / period_nm)
        crystal.set_device(["grating", "waveguide"], fields_mask=False)
        polarization = str(configuration.parameter("polarization"))
        te, tm = ((1.0, 0.0) if polarization == "te" else (0.0, 1.0))
        wavelength_nm = float(configuration.parameter("wavelength_nm"))
        crystal.set_source(
            wavelength_nm / period_nm,
            te=te,
            tm=tm,
            theta=1e-6,
            phi=0.0,
        )
        crystal.solve()
        reflection, transmission, absorption = _rta(crystal)
        return CaseResult(
            {"R": reflection, "T": transmission, "A": absorption},
            metadata={"wavelength_nm": wavelength_nm},
            retained_bytes=crystal.Stot.nbytes,
        )

    def collect(self, evaluations: Sequence[CaseEvaluation]) -> CaseResult:
        evaluations = tuple(
            sorted(evaluations, key=lambda item: item.configuration.ordinal)
        )
        wavelengths = np.asarray(
            [item.configuration.parameter("wavelength_nm") for item in evaluations],
            dtype=float,
        )
        transmission = np.asarray(
            [item.result.observables["T"] for item in evaluations], dtype=float
        )
        energy_defect = np.asarray(
            [abs(item.result.observables["A"]) for item in evaluations], dtype=float
        )
        resonance_index = int(np.argmin(transmission))
        return CaseResult(
            {
                "resonance_nm": float(wavelengths[resonance_index]),
                "minimum_T": float(transmission[resonance_index]),
                "max_energy_defect": float(np.max(energy_defect)),
            },
            series={"wavelength_nm": wavelengths, "T": transmission},
            metadata={"fixture": "luder_2020/manifest.json"},
        )

    def aggregate_reference(self, aggregate: CaseResult) -> ReferenceResult:
        return ReferenceResult(
            {"resonance_nm": self.expected_nm},
            "graphical-literature-scalar",
            self.references[0],
            fixture="luder_2020/manifest.json",
            uncertainty=str(self.manifest["uncertainty"]),
        )


@dataclass(frozen=True)
class LouTwistMapCase(ValidationCase):
    """Lou frequency/twist map with one independent map pixel per task."""

    frequencies: Tuple[float, ...] = (0.74, 0.78)
    twists_degrees: Tuple[float, ...] = (5.0, 15.0)
    pw: PlaneWaves = (1, 1)
    polarization: str = "rcp"

    slug = "lou-twist-map"
    title = "Lou twisted-bilayer R/T map"
    description = "Cartesian product of frequencies and twists; each pixel is independent."
    tags = ("extended", "literature", "map", "parallel")
    references = (
        LiteratureReference(
            "B. Lou et al., Phys. Rev. Lett. 126, 136101 (2021)",
            "https://doi.org/10.1103/PhysRevLett.126.136101",
            "Two perforated epsilon=4 slabs separated by a 0.3a air gap.",
        ),
    )

    def configurations(self) -> Iterable[CaseConfiguration]:
        # Frequency-major order makes collected arrays naturally [frequency, twist].
        for ordinal, (frequency, twist) in enumerate(
            product(self.frequencies, self.twists_degrees)
        ):
            yield CaseConfiguration.create(
                self.slug,
                f"frequency={frequency:.12g};twist={twist:.12g}",
                ordinal,
                self.pw,
                {
                    "frequency_c_over_a": frequency,
                    "twist_degrees": twist,
                    "polarization": self.polarization.lower(),
                },
            )

    @staticmethod
    def _polarization(name: str) -> Tuple[complex, complex]:
        values = {
            "te": (1.0, 0.0),
            "tm": (0.0, 1.0),
            "rcp": (1.0, 1.0j),
            "lcp": (1.0, -1.0j),
        }
        try:
            return values[name]
        except KeyError as error:
            raise ValueError(f"unknown polarization {name!r}") from error

    def evaluate(self, configuration: CaseConfiguration) -> CaseResult:
        frequency = float(configuration.parameter("frequency_c_over_a"))
        twist = float(configuration.parameter("twist_degrees"))
        pattern = Drawing((8, 8), 4.0)
        pattern.disc((0.0, 0.0), 0.25, 1.0)
        first = Expansion(configuration.pw)
        second = Expansion(configuration.pw)
        second.rotate(radians(twist))
        extended = first + second
        crystal = Crystal.from_expansion(extended)
        crystal.add_layer(
            "upper",
            Layer.analytical(first, pattern.islands(), 4.0, 0.2),
            extended=True,
        )
        crystal.add_layer("gap", Layer.uniform(extended, 1.0, 0.3))
        crystal.add_layer(
            "lower",
            Layer.analytical(second, pattern.islands(), 4.0, 0.2),
            extended=True,
        )
        crystal.set_device(["upper", "gap", "lower"], fields_mask=False)
        te, tm = self._polarization(str(configuration.parameter("polarization")))
        crystal.set_source(1.0 / frequency, te=te, tm=tm)
        crystal.solve()
        reflection, transmission, absorption = _rta(crystal)
        return CaseResult(
            {"R": reflection, "T": transmission, "A": absorption},
            metadata={
                "frequency_c_over_a": frequency,
                "twist_degrees": twist,
                "extended_harmonics": int(np.prod(extended.pw)),
            },
            retained_bytes=crystal.Stot.nbytes,
        )

    def collect(self, evaluations: Sequence[CaseEvaluation]) -> CaseResult:
        expected = len(self.frequencies) * len(self.twists_degrees)
        if len(evaluations) != expected:
            raise ValueError(f"expected {expected} map pixels, received {len(evaluations)}")
        evaluations = tuple(
            sorted(evaluations, key=lambda item: item.configuration.ordinal)
        )
        shape = (len(self.frequencies), len(self.twists_degrees))
        series = {
            name: np.asarray(
                [item.result.observables[name] for item in evaluations], dtype=float
            ).reshape(shape)
            for name in ("R", "T", "A")
        }
        series["frequencies"] = np.asarray(self.frequencies, dtype=float)
        series["twists_degrees"] = np.asarray(self.twists_degrees, dtype=float)
        return CaseResult(
            {
                "max_energy_defect": float(np.max(np.abs(series["A"]))),
                "maximum_T": float(np.max(series["T"])),
            },
            series=series,
            metadata={
                "axis_order": ["frequency_c_over_a", "twist_degrees"],
                "fixture": "lou_2021/manifest.json",
            },
        )


def case_registry():
    """Return fresh default case instances, keyed by stable public slug."""

    cases = (
        ThinFilmCase(),
        BrewsterInterfaceCase(),
        GuidedModeSpectrumCase(),
        LouTwistMapCase(),
    )
    return {case.slug: case for case in cases}
