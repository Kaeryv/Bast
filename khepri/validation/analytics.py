"""Small analytical references independent from Khepri's RCWA implementation."""

from cmath import cos, sin, sqrt
from math import atan

import numpy as np


def brewster_angle(epsilon_incident, epsilon_transmitted):
    return atan(sqrt(epsilon_transmitted / epsilon_incident).real)


def fresnel_interface(epsilon_incident, epsilon_transmitted, theta, polarization):
    n0 = sqrt(epsilon_incident)
    n1 = sqrt(epsilon_transmitted)
    sin_transmitted = n0 * sin(theta) / n1
    cos_incident = cos(theta)
    cos_transmitted = sqrt(1 - sin_transmitted**2)
    if polarization.lower() in ("s", "te"):
        reflection_amplitude = (
            n0 * cos_incident - n1 * cos_transmitted
        ) / (n0 * cos_incident + n1 * cos_transmitted)
    else:
        reflection_amplitude = (
            n1 * cos_incident - n0 * cos_transmitted
        ) / (n1 * cos_incident + n0 * cos_transmitted)
    reflection = abs(reflection_amplitude) ** 2
    return {"R": float(reflection), "T": float(1 - reflection), "A": 0.0}


def single_film(
    epsilon_incident,
    epsilon_film,
    epsilon_substrate,
    thickness,
    wavelength,
    theta,
    polarization,
):
    n0 = sqrt(epsilon_incident)
    n1 = sqrt(epsilon_film)
    n2 = sqrt(epsilon_substrate)
    s0 = sin(theta)
    c0 = cos(theta)
    s1 = n0 * s0 / n1
    s2 = n0 * s0 / n2
    c1 = sqrt(1 - s1**2)
    c2 = sqrt(1 - s2**2)
    if polarization.lower() in ("s", "te"):
        q0, q1, q2 = n0 * c0, n1 * c1, n2 * c2
    else:
        q0, q1, q2 = c0 / n0, c1 / n1, c2 / n2
    r01 = (q0 - q1) / (q0 + q1)
    r12 = (q1 - q2) / (q1 + q2)
    phase = 2 * 3.141592653589793 * n1 * c1 * thickness / wavelength
    from cmath import exp

    reflection_amplitude = (r01 + r12 * exp(2j * phase)) / (
        1 + r01 * r12 * exp(2j * phase)
    )
    reflection = abs(reflection_amplitude) ** 2
    transmission = 1 - reflection
    return {"R": float(reflection), "T": float(transmission), "A": 0.0}


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
