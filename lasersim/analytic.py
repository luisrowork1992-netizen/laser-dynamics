"""Closed-form results for Q-switched pulses and laser cavities."""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "C_LIGHT",
    "qswitch_peak_photons",
    "qswitch_final_inversion",
    "qswitch_extraction_efficiency",
    "photon_lifetime",
    "threshold_gain",
    "mode_spacing",
    "rigrod_output",
    "optimal_output_coupling",
]

C_LIGHT = 299_792_458.0  # m/s


def qswitch_peak_photons(initial_inversion: float, threshold_inversion: float) -> float:
    """Peak photon density of an ideal Q-switched pulse.

    ``S_peak = N_i - N_th - N_th ln(N_i / N_th)`` (four-level laser, pump and
    spontaneous decay neglected during the pulse).
    """
    r = initial_inversion / threshold_inversion
    if r <= 1:
        return 0.0
    return threshold_inversion * (r - 1 - np.log(r))


def qswitch_final_inversion(initial_inversion: float, threshold_inversion: float) -> float:
    """Inversion left after the pulse: root of ``N_i - N_f = N_th ln(N_i / N_f)`` below ``N_th``."""
    Ni, Nth = initial_inversion, threshold_inversion
    if Ni <= Nth:
        return Ni
    f = lambda Nf: Ni - Nf - Nth * np.log(Ni / Nf)
    return float(brentq(f, Ni * 1e-15, Nth * (1 - 1e-12), xtol=1e-14 * Ni))


def qswitch_extraction_efficiency(initial_inversion: float, threshold_inversion: float) -> float:
    """Fraction of the stored inversion converted to photons, ``1 - N_f / N_i``."""
    return 1.0 - qswitch_final_inversion(initial_inversion, threshold_inversion) / initial_inversion


def photon_lifetime(length: float, R1: float, R2: float, loss_per_length: float = 0.0,
                    n: float = 1.0) -> float:
    """Cavity photon lifetime ``2 n L / (c [-ln(R1 R2) + 2 alpha L])`` (SI units)."""
    round_trip_loss = -np.log(R1 * R2) + 2 * loss_per_length * length
    if round_trip_loss <= 0:
        raise ValueError("cavity has no loss")
    return 2 * n * length / (C_LIGHT * round_trip_loss)


def threshold_gain(length: float, R1: float, R2: float, loss_per_length: float = 0.0) -> float:
    """Threshold gain coefficient ``alpha + ln(1 / (R1 R2)) / (2 L)``."""
    return loss_per_length + np.log(1.0 / (R1 * R2)) / (2 * length)


def mode_spacing(length: float, n: float = 1.0) -> float:
    """Longitudinal-mode (free spectral range) spacing ``c / (2 n L)`` in Hz."""
    return C_LIGHT / (2 * n * length)


def rigrod_output(T, g0l: float, internal_loss: float):
    """Low-gain (Rigrod) output power, in units of the saturation power.

    ``P_out = T (2 g0 l / (T + L) - 1) / 2`` for output-coupler transmission ``T``,
    single-pass small-signal gain ``g0 l`` and round-trip internal loss ``L``.
    Negative values (below threshold) are clipped to zero.
    """
    T = np.asarray(T, dtype=float)
    p = 0.5 * T * (2 * g0l / (T + internal_loss) - 1)
    return np.clip(p, 0.0, None)


def optimal_output_coupling(g0l: float, internal_loss: float) -> float:
    """Transmission that maximises :func:`rigrod_output`: ``sqrt(2 g0 l L) - L``."""
    t = np.sqrt(2 * g0l * internal_loss) - internal_loss
    return float(max(t, 0.0))
