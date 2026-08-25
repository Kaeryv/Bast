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

from .analytics import (
    brewster_angle,
    fresnel_interface,
    multilayer_stack,
    single_film,
)
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
    references = (
        LiteratureReference(
            "M. Born and E. Wolf, Principles of Optics, 7th ed.",
            "https://doi.org/10.1017/CBO9781139644187",
            "Fresnel coefficients and the TM Brewster zero.",
        ),
    )

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
        return ReferenceResult(values, "analytical", self.references[0])


@dataclass(frozen=True)
class FabryPerotCavityCase(ValidationCase):
    """Two finite dielectric mirrors separated by a variable air cavity."""

    pw: PlaneWaves = (1, 1)
    wavelengths: Tuple[float, ...] = (0.9, 1.0, 1.1)
    gaps: Tuple[float, ...] = (0.20, 0.50, 0.80)
    polarizations: Tuple[str, ...] = ("s", "p")
    epsilon_incident: float = 1.0
    epsilon_mirror: float = 4.0
    epsilon_cavity: float = 1.0
    epsilon_substrate: float = 1.0
    mirror_thickness: float = 0.125

    slug = "fabry-perot"
    title = "Fabry–Pérot cavity between finite dielectric mirrors"
    description = (
        "Two identical dielectric films separated by a variable cavity, "
        "checked against an independent characteristic matrix."
    )
    tags = ("fast", "analytic", "integration", "fabry-perot", "parallel")
    references = (
        LiteratureReference(
            "H. A. Macleod, Thin-Film Optical Filters, 4th ed.",
            "https://doi.org/10.1201/b21960",
            "Characteristic matrices and Fabry–Pérot multiple reflection.",
        ),
    )

    def configurations(self) -> Iterable[CaseConfiguration]:
        for ordinal, (polarization, wavelength, gap) in enumerate(
            product(self.polarizations, self.wavelengths, self.gaps)
        ):
            yield CaseConfiguration.create(
                self.slug,
                (
                    f"pol={polarization.lower()};wavelength={wavelength:.12g};"
                    f"gap={gap:.12g}"
                ),
                ordinal,
                self.pw,
                {
                    "polarization": polarization.lower(),
                    "wavelength": wavelength,
                    "gap": gap,
                },
            )

    def evaluate(self, configuration: CaseConfiguration) -> CaseResult:
        polarization = str(configuration.parameter("polarization"))
        wavelength = float(configuration.parameter("wavelength"))
        gap = float(configuration.parameter("gap"))
        crystal = Crystal(
            configuration.pw,
            lattice_pitch=0.37,
            epsi=self.epsilon_incident,
            epse=self.epsilon_substrate,
        )
        crystal.add_layer_uniform(
            "mirror", self.epsilon_mirror, self.mirror_thickness
        )
        crystal.add_layer_uniform("cavity", self.epsilon_cavity, gap)
        crystal.set_device(["mirror", "cavity", "mirror"], fields_mask=False)
        te, tm = ((1.0, 0.0) if polarization in ("s", "te") else (0.0, 1.0))
        crystal.set_source(wavelength, te=te, tm=tm)
        crystal.solve()
        reflection, transmission, absorption = _rta(crystal)
        return CaseResult(
            {"R": reflection, "T": transmission, "A": absorption},
            metadata={"gap": gap, "wavelength": wavelength},
            retained_bytes=crystal.Stot.nbytes,
        )

    def reference(self, configuration: CaseConfiguration) -> ReferenceResult:
        gap = float(configuration.parameter("gap"))
        values = multilayer_stack(
            self.epsilon_incident,
            self.epsilon_substrate,
            (
                (self.epsilon_mirror, self.mirror_thickness),
                (self.epsilon_cavity, gap),
                (self.epsilon_mirror, self.mirror_thickness),
            ),
            float(configuration.parameter("wavelength")),
        )
        return ReferenceResult(values, "analytical", self.references[0])

    def collect(self, evaluations: Sequence[CaseEvaluation]) -> CaseResult:
        expected = len(self.polarizations) * len(self.wavelengths) * len(self.gaps)
        if len(evaluations) != expected:
            raise ValueError(
                f"expected {expected} cavity points, received {len(evaluations)}"
            )
        evaluations = tuple(
            sorted(evaluations, key=lambda item: item.configuration.ordinal)
        )
        shape = (len(self.polarizations), len(self.wavelengths), len(self.gaps))
        return CaseResult(
            {
                "max_energy_defect": float(
                    max(abs(item.result.observables["A"]) for item in evaluations)
                )
            },
            series={
                name: np.asarray(
                    [item.result.observables[name] for item in evaluations]
                ).reshape(shape)
                for name in ("R", "T", "A")
            },
            metadata={
                "axis_order": ["polarization", "wavelength", "gap"],
                "polarizations": list(self.polarizations),
                "wavelengths": list(self.wavelengths),
                "gaps": list(self.gaps),
            },
        )


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
    backend: str = "dense"

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
                    "backend": self.backend,
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
        crystal.solve(backend=str(configuration.parameter("backend")))
        reflection, transmission, absorption = _rta(crystal)
        actual_backend = crystal.solve_diagnostics["backend"]
        retained_bytes = (
            crystal.solve_diagnostics["retained_bytes"]
            if actual_backend == "operator"
            else crystal.Stot.nbytes
        )
        return CaseResult(
            {"R": reflection, "T": transmission, "A": absorption},
            metadata={
                "frequency_c_over_a": frequency,
                "twist_degrees": twist,
                "extended_harmonics": int(np.prod(extended.pw)),
                "backend": actual_backend,
            },
            retained_bytes=retained_bytes,
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


@dataclass(frozen=True)
class DisplacementSensitiveSlabCase(ValidationCase):
    """Longitudinal and lateral displacement of two guided-resonant slabs."""

    pw: PlaneWaves = (5, 5)
    frequency_range: Tuple[float, float] = (0.49, 0.59)
    samples: int = 101
    gaps: Tuple[float, ...] = (1.35, 1.10, 0.95, 0.85, 0.75, 0.65, 0.55)
    lateral_shifts: Tuple[float, ...] = (0.0,)
    polarization: str = "te"
    slab_epsilon: float = 12.0
    hole_epsilon: float = 1.0
    hole_radius: float = 0.4
    slab_thickness: float = 0.55

    slug = "fan-displacement-sensitive-slabs"
    title = "Fan displacement-sensitive coupled photonic-crystal slabs"
    description = (
        "Two square-hole photonic-crystal slabs with independently sampled "
        "frequency, longitudinal gap, and lateral registry."
    )
    tags = (
        "slow",
        "literature",
        "guided-resonance",
        "displacement",
        "spectrum",
        "parallel",
    )
    references = (
        LiteratureReference(
            (
                "W. Suh, M. F. Yanik, O. Solgaard, and S. Fan, "
                "Appl. Phys. Lett. 82, 1999–2001 (2003)"
            ),
            "https://doi.org/10.1063/1.1563739",
            "Figures 2 and 4: longitudinal and lateral displacement sensitivity.",
        ),
    )

    @property
    def manifest(self):
        return load_manifest("suh_fan_2003/manifest.json")

    @classmethod
    def published_lateral_study(cls, **overrides):
        """Build the paper's h=0.1a, shift=(0, 0.05a) comparison."""

        parameters = {
            "gaps": (0.1,),
            "lateral_shifts": (0.0, 0.05),
        }
        parameters.update(overrides)
        return cls(**parameters)

    def frequencies(self) -> np.ndarray:
        if self.samples < 2:
            raise ValueError("samples must be at least two")
        if self.frequency_range[0] >= self.frequency_range[1]:
            raise ValueError("frequency_range must be strictly increasing")
        return np.linspace(
            self.frequency_range[0], self.frequency_range[1], self.samples
        )

    def configurations(self) -> Iterable[CaseConfiguration]:
        for ordinal, (gap, shift, frequency) in enumerate(
            product(self.gaps, self.lateral_shifts, self.frequencies())
        ):
            yield CaseConfiguration.create(
                self.slug,
                (
                    f"gap={gap:.12g};shift_x={shift:.12g};"
                    f"frequency={frequency:.12g}"
                ),
                ordinal,
                self.pw,
                {
                    "gap_over_a": gap,
                    "shift_x_over_a": shift,
                    "frequency_c_over_a": float(frequency),
                    "polarization": self.polarization.lower(),
                },
            )

    def _patterned_layer(self, expansion, shift_x):
        drawing = Drawing((8, 8), self.slab_epsilon)
        drawing.disc((shift_x, 0.0), self.hole_radius, self.hole_epsilon)
        return Layer.analytical(
            expansion,
            drawing.islands(),
            drawing.background,
            self.slab_thickness,
        )

    def evaluate(self, configuration: CaseConfiguration) -> CaseResult:
        gap = float(configuration.parameter("gap_over_a"))
        shift = float(configuration.parameter("shift_x_over_a"))
        frequency = float(configuration.parameter("frequency_c_over_a"))
        expansion = Expansion(configuration.pw)
        crystal = Crystal.from_expansion(expansion, epsi=1.0, epse=1.0)
        crystal.add_layer("upper", self._patterned_layer(expansion, 0.0))
        crystal.add_layer_uniform("gap", 1.0, gap)
        crystal.add_layer("lower", self._patterned_layer(expansion, shift))
        crystal.set_device(["upper", "gap", "lower"], fields_mask=False)
        polarization = str(configuration.parameter("polarization"))
        te, tm = ((1.0, 0.0) if polarization in ("s", "te") else (0.0, 1.0))
        crystal.set_source(1.0 / frequency, te=te, tm=tm)
        crystal.solve()
        reflection, transmission, absorption = _rta(crystal)
        return CaseResult(
            {"R": reflection, "T": transmission, "A": absorption},
            metadata={
                "frequency_c_over_a": frequency,
                "gap_over_a": gap,
                "shift_x_over_a": shift,
            },
            retained_bytes=crystal.Stot.nbytes,
        )

    def collect(self, evaluations: Sequence[CaseEvaluation]) -> CaseResult:
        expected = len(self.gaps) * len(self.lateral_shifts) * self.samples
        if len(evaluations) != expected:
            raise ValueError(
                f"expected {expected} spectrum pixels, received {len(evaluations)}"
            )
        evaluations = tuple(
            sorted(evaluations, key=lambda item: item.configuration.ordinal)
        )
        shape = (len(self.gaps), len(self.lateral_shifts), self.samples)
        transmission = np.asarray(
            [item.result.observables["T"] for item in evaluations]
        ).reshape(shape)
        reflection = np.asarray(
            [item.result.observables["R"] for item in evaluations]
        ).reshape(shape)
        absorption = np.asarray(
            [item.result.observables["A"] for item in evaluations]
        ).reshape(shape)
        frequencies = self.frequencies()
        return CaseResult(
            {
                "max_energy_defect": float(np.max(np.abs(absorption))),
                "maximum_T": float(np.max(transmission)),
            },
            series={
                "frequencies": frequencies,
                "gaps": np.asarray(self.gaps),
                "lateral_shifts": np.asarray(self.lateral_shifts),
                "R": reflection,
                "T": transmission,
                "A": absorption,
            },
            metadata={
                "axis_order": ["gap_over_a", "shift_x_over_a", "frequency_c_over_a"],
                "fixture": "suh_fan_2003/manifest.json",
                "feature_extraction": (
                    "none: narrow resonances require adaptive frequency refinement "
                    "and plane-wave convergence"
                ),
                "reference_scope": (
                    "graphical qualitative trends; no digitized curve bundled"
                ),
            },
        )


def case_registry():
    """Return fresh default case instances, keyed by stable public slug."""

    cases = (
        ThinFilmCase(),
        BrewsterInterfaceCase(),
        FabryPerotCavityCase(),
        GuidedModeSpectrumCase(),
        LouTwistMapCase(),
        DisplacementSensitiveSlabCase(),
    )
    return {case.slug: case for case in cases}
