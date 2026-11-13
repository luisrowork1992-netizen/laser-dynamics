"""Four-level laser rate equations.

State: inversion density ``N`` and intracavity photon density ``S`` (any
consistent units, for example cm^-3 and seconds)::

    dN/dt = P - N / tau - G N S
    dS/dt = G N S - S / tau_p + beta N / tau

* ``P``     pump rate (inversions created per unit volume per unit time)
* ``tau``   upper-level (spontaneous) lifetime
* ``tau_p`` cavity photon lifetime
* ``G``     stimulated-emission coefficient (sigma * c / n)
* ``beta``  fraction of spontaneous emission that seeds the lasing mode
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import find_peaks

__all__ = ["FourLevelLaser", "Trace"]


@dataclass
class Trace:
    t: np.ndarray
    N: np.ndarray
    S: np.ndarray

    def output(self, tau_p: float, coupling_fraction: float = 1.0) -> np.ndarray:
        """Photon flux leaving the cavity, ``coupling_fraction * S / tau_p``."""
        return coupling_fraction * self.S / tau_p


@dataclass(frozen=True)
class FourLevelLaser:
    tau: float
    tau_p: float
    G: float
    beta: float = 0.0

    def __post_init__(self):
        if min(self.tau, self.tau_p, self.G) <= 0:
            raise ValueError("tau, tau_p and G must be positive")
        if not 0 <= self.beta < 1:
            raise ValueError("beta must be in [0, 1)")

    # --- steady state -------------------------------------------------
    @property
    def N_th(self) -> float:
        """Threshold (clamped) inversion, ``1 / (G tau_p)``."""
        return 1.0 / (self.G * self.tau_p)

    @property
    def P_th(self) -> float:
        """Threshold pump rate, ``N_th / tau``."""
        return self.N_th / self.tau

    def steady_state(self, P: float):
        """Exact steady state ``(N, S)`` including spontaneous seeding (``beta``)."""
        if P < 0:
            raise ValueError("pump rate must be non-negative")
        if self.beta == 0.0:
            if P <= self.P_th:
                return P * self.tau, 0.0
            return self.N_th, self.tau_p * (P - self.P_th)
        # with beta > 0: eliminate N from dS/dt = 0, then solve a quadratic in S
        # N = (S / tau_p) / (G S + beta / tau);  substitute into dN/dt = 0
        g, tp, t, b = self.G, self.tau_p, self.tau, self.beta
        # P (G S + b/t) - (S/tp)(1/t + G S) = 0  ->  -(G/tp) S^2 + (P G - 1/(tp t)) S + P b / t = 0
        A, B, C = -g / tp, P * g - 1.0 / (tp * t), P * b / t
        S = (-B - np.sqrt(B * B - 4 * A * C)) / (2 * A)
        N = (S / tp) / (g * S + b / t)
        return float(N), float(S)

    def relaxation_oscillation(self, P: float):
        """Small-signal relaxation-oscillation angular frequency and damping rate.

        With ``r = P / P_th``: ``omega_R = sqrt((r - 1) / (tau tau_p))`` and
        ``gamma = r / (2 tau)``. The damped frequency is ``sqrt(omega_R^2 - gamma^2)``.
        """
        r = P / self.P_th
        if r <= 1:
            raise ValueError("relaxation oscillations need pumping above threshold")
        omega_r = np.sqrt((r - 1) / (self.tau * self.tau_p))
        gamma = r / (2 * self.tau)
        return float(omega_r), float(gamma)

    # --- dynamics -----------------------------------------------------
    def _rhs(self, pump):
        def f(t, y):
            N, S = y
            P = pump(t) if callable(pump) else pump
            gain = self.G * N * S
            return [P - N / self.tau - gain, gain - S / self.tau_p + self.beta * N / self.tau]
        return f

    def _rhs_log(self, pump):
        # same equations with u = ln S: keeps S positive and tames the huge
        # dynamic range of the turn-on spikes (S grows from ~1e-12 to ~1e3)
        def f(t, y):
            N, u = y
            P = pump(t) if callable(pump) else pump
            S = np.exp(u)
            dN = P - N / self.tau - self.G * N * S
            du = self.G * N - 1.0 / self.tau_p
            if self.beta > 0:
                du += self.beta * N / (self.tau * S)
            return [dN, du]
        return f

    def simulate(self, pump, t_end: float, N0: float = 0.0, S0: float = 1e-12,
                 n_points: int = 4000, rtol: float = 1e-10, atol: float = 1e-10) -> Trace:
        """Integrate the rate equations. ``pump`` may be a constant or a function of time.

        Internally the photon density is integrated as ``ln S``, so ``S0`` must be > 0.
        """
        if S0 <= 0:
            raise ValueError("S0 must be positive (use a tiny seed such as 1e-12)")
        t_eval = np.linspace(0.0, t_end, n_points)
        scale = max(abs(N0), self.N_th)
        sol = solve_ivp(self._rhs_log(pump), (0.0, t_end), [N0, np.log(S0)], method="Radau",
                        t_eval=t_eval, rtol=rtol, atol=[atol * scale, atol])
        if not sol.success:
            raise RuntimeError(sol.message)
        return Trace(sol.t, sol.y[0], np.exp(sol.y[1]))

    def measured_oscillation_frequency(self, trace: Trace) -> float:
        """Angular frequency of the ringing in ``S``, from the spacing of its later peaks.

        The first spikes after turn-on are strongly nonlinear and slower, so only the
        second half of the peaks (the small-signal ringing) is used.
        """
        peaks, _ = find_peaks(trace.S)
        late = peaks[peaks.size // 2:]
        if late.size < 3:
            raise ValueError("not enough oscillation peaks in the trace")
        period = np.mean(np.diff(trace.t[late]))
        return float(2 * np.pi / period)

    # --- Q-switching ---------------------------------------------------
    def q_switch(self, initial_inversion: float, t_end: float, seed: float = 1e-9,
                 n_points: int = 20000) -> Trace:
        """Giant-pulse emission from a stored inversion, with the pump off."""
        return self.simulate(0.0, t_end, N0=initial_inversion, S0=seed, n_points=n_points)
