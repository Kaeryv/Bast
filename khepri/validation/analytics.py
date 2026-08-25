"""Small analytical references independent from Khepri's RCWA implementation."""

from cmath import cos, sin, sqrt
from math import atan

import numpy as np


def _forward_cosine(index, transverse_index):
    """Return the passive forward-wave cosine for Khepri's exp(+iwt) convention."""

    value = sqrt(1 - (transverse_index / index) ** 2)
    longitudinal_index = index * value
    if longitudinal_index.real < 0 or (
        abs(longitudinal_index.real) < 1e-14
        and longitudinal_index.imag > 0
    ):
        value = -value
    return value


def _optical_admittance(index, cosine, polarization):
    if polarization.lower() in ("s", "te"):
        return index * cosine
    if polarization.lower() in ("p", "tm"):
        return cosine / index
    raise ValueError(f"unknown polarization {polarization!r}")


def brewster_angle(epsilon_incident, epsilon_transmitted):
    return atan(sqrt(epsilon_transmitted / epsilon_incident).real)


def fresnel_interface(epsilon_incident, epsilon_transmitted, theta, polarization):
    n0 = sqrt(epsilon_incident)
    n1 = sqrt(epsilon_transmitted)
    transverse_index = n0 * sin(theta)
    cos_incident = cos(theta)
    cos_transmitted = _forward_cosine(n1, transverse_index)
    q0 = _optical_admittance(n0, cos_incident, polarization)
    q1 = _optical_admittance(n1, cos_transmitted, polarization)
    reflection_amplitude = (q0 - q1) / (q0 + q1)
    transmission_amplitude = 2 * q0 / (q0 + q1)
    reflection = abs(reflection_amplitude) ** 2
    transmission = np.real(q1) / np.real(q0) * abs(transmission_amplitude) ** 2
    return {
        "R": float(np.real(reflection)),
        "T": float(np.real(transmission)),
        "A": 0.0,
    }


def single_film(
    epsilon_incident,
    epsilon_film,
    epsilon_substrate,
    thickness,
    wavelength,
    theta,
    polarization,
):
    return multilayer_stack_oblique(
        epsilon_incident,
        epsilon_substrate,
        ((epsilon_film, thickness),),
        wavelength,
        theta,
        polarization,
    )


def multilayer_stack_oblique(
    epsilon_incident,
    epsilon_substrate,
    layers,
    wavelength,
    theta,
    polarization,
):
    """Oblique characteristic-matrix solution, including passive loss.

    Khepri uses an ``exp(+i omega t)`` convention, so passive refractive
    indices and permittivities have negative imaginary parts.  The exterior
    media should be lossless when interpreting ``A`` as absorption inside the
    finite layer sequence.
    """

    wavelength = complex(wavelength)
    if wavelength == 0:
        raise ValueError("wavelength must be non-zero")
    n0 = sqrt(complex(epsilon_incident))
    ns = sqrt(complex(epsilon_substrate))
    transverse_index = n0 * sin(theta)
    c0 = cos(theta)
    cs = _forward_cosine(ns, transverse_index)
    q0 = _optical_admittance(n0, c0, polarization)
    qs = _optical_admittance(ns, cs, polarization)

    total = np.eye(2, dtype=np.complex128)
    for epsilon, thickness in layers:
        index = sqrt(complex(epsilon))
        cosine = _forward_cosine(index, transverse_index)
        admittance = _optical_admittance(index, cosine, polarization)
        phase = 2 * np.pi * index * cosine * thickness / wavelength
        layer = np.asarray(
            (
                (np.cos(phase), 1j * np.sin(phase) / admittance),
                (1j * admittance * np.sin(phase), np.cos(phase)),
            ),
            dtype=np.complex128,
        )
        total = total @ layer

    a, b = total[0]
    c, d = total[1]
    denominator = q0 * (a + b * qs) + c + d * qs
    reflection_amplitude = (q0 * (a + b * qs) - c - d * qs) / denominator
    transmission_amplitude = 2 * q0 / denominator
    reflection = abs(reflection_amplitude) ** 2
    transmission = np.real(qs) / np.real(q0) * abs(transmission_amplitude) ** 2
    absorption = 1.0 - reflection - transmission
    return {
        "R": float(np.real(reflection)),
        "T": float(np.real(transmission)),
        "A": float(np.real(absorption)),
    }


def multilayer_stack(
    epsilon_incident,
    epsilon_substrate,
    layers,
    wavelength,
):
    """Normal-incidence characteristic-matrix solution for finite layers.

    ``layers`` is an incidence-to-transmission sequence of
    ``(epsilon, thickness)`` pairs.  The calculation is independent from
    Khepri's scattering matrices and therefore provides a useful stacking and
    multiple-reflection oracle.
    """

    wavelength = complex(wavelength)
    if wavelength == 0:
        raise ValueError("wavelength must be non-zero")
    total = np.eye(2, dtype=np.complex128)
    for epsilon, thickness in layers:
        index = np.sqrt(complex(epsilon))
        phase = 2 * np.pi * index * thickness / wavelength
        layer = np.asarray(
            (
                (np.cos(phase), 1j * np.sin(phase) / index),
                (1j * index * np.sin(phase), np.cos(phase)),
            ),
            dtype=np.complex128,
        )
        total = total @ layer

    q0 = np.sqrt(complex(epsilon_incident))
    qs = np.sqrt(complex(epsilon_substrate))
    a, b = total[0]
    c, d = total[1]
    denominator = q0 * (a + b * qs) + c + d * qs
    reflection_amplitude = (
        q0 * (a + b * qs) - c - d * qs
    ) / denominator
    transmission_amplitude = 2 * q0 / denominator
    reflection = abs(reflection_amplitude) ** 2
    transmission = (
        np.real(qs) / np.real(q0) * abs(transmission_amplitude) ** 2
    )
    absorption = 1.0 - reflection - transmission
    return {
        "R": float(np.real(reflection)),
        "T": float(np.real(transmission)),
        "A": float(np.real(absorption)),
    }
