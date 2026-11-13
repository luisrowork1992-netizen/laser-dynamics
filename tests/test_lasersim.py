import numpy as np
import pytest

import lasersim as ls

# normalised units: lifetimes in units of tau, densities in units of 1/(G tau)
LASER = ls.FourLevelLaser(tau=1.0, tau_p=1e-3, G=1.0)


def test_threshold_values():
    assert LASER.N_th == pytest.approx(1000.0)
    assert LASER.P_th == pytest.approx(1000.0)


@pytest.mark.parametrize("r", [0.5, 1.5, 3.0])
def test_simulation_settles_to_steady_state(r):
    P = r * LASER.P_th
    tr = LASER.simulate(P, t_end=40.0)  # spiking near threshold damps slowly
    N_ss, S_ss = LASER.steady_state(P)
    assert tr.N[-1] == pytest.approx(N_ss, rel=1e-4)
    if r > 1:
        assert tr.S[-1] == pytest.approx(S_ss, rel=1e-4)
        assert N_ss == pytest.approx(LASER.N_th)  # gain clamping
    else:
        assert tr.S[-1] < 1e-6


def test_output_is_linear_above_threshold():
    S = [LASER.steady_state(r * LASER.P_th)[1] for r in (2.0, 3.0, 4.0)]
    slope = np.diff(S) / LASER.P_th
    np.testing.assert_allclose(slope, LASER.tau_p)


def test_steady_state_with_spontaneous_seeding():
    laser = ls.FourLevelLaser(tau=1.0, tau_p=1e-3, G=1.0, beta=1e-4)
    for P in (500.0, 1000.0, 2500.0):
        N, S = laser.steady_state(P)
        dN, dS = laser._rhs(P)(0.0, [N, S])
        assert abs(dN) < 1e-9 * P and abs(dS) < 1e-9 * P
        assert S > 0
    # far above threshold the seeded result approaches the ideal one
    N, S = laser.steady_state(3000.0)
    assert S == pytest.approx(LASER.steady_state(3000.0)[1], rel=1e-3)


def test_relaxation_oscillation_frequency():
    P = 2.0 * LASER.P_th
    omega_r, gamma = LASER.relaxation_oscillation(P)
    assert omega_r == pytest.approx(np.sqrt(1 / 1e-3))
    assert gamma == pytest.approx(1.0)
    tr = LASER.simulate(P, t_end=8.0, N0=0.0, S0=1e-6, n_points=8000)
    measured = LASER.measured_oscillation_frequency(tr)
    assert measured == pytest.approx(np.sqrt(omega_r**2 - gamma**2), rel=1e-2)


def test_relaxation_needs_lasing():
    with pytest.raises(ValueError):
        LASER.relaxation_oscillation(0.5 * LASER.P_th)


@pytest.mark.parametrize("r", [1.5, 3.0, 6.0])
def test_q_switched_pulse_matches_theory(r):
    laser = ls.FourLevelLaser(tau=1e6, tau_p=1e-3, G=1.0)  # spontaneous decay negligible
    Ni = r * laser.N_th
    tr = laser.q_switch(Ni, t_end=0.2 if r < 2 else 0.06, n_points=60000)
    assert tr.S.max() == pytest.approx(ls.qswitch_peak_photons(Ni, laser.N_th), rel=2e-3)
    assert tr.N[-1] == pytest.approx(ls.qswitch_final_inversion(Ni, laser.N_th), rel=2e-3)
    # the inversion at the pulse peak equals the threshold inversion
    assert tr.N[np.argmax(tr.S)] == pytest.approx(laser.N_th, rel=5e-3)


def test_extraction_efficiency_equation():
    for r in (1.2, 2.0, 5.0):
        eta = ls.qswitch_extraction_efficiency(r, 1.0)
        assert r * eta == pytest.approx(-np.log(1 - eta), rel=1e-9)
    assert ls.qswitch_peak_photons(0.8, 1.0) == 0.0


def test_cavity_formulas():
    assert ls.mode_spacing(1.0) == pytest.approx(149.896229e6, rel=1e-8)
    assert ls.threshold_gain(0.5, 1.0, 0.9) == pytest.approx(-np.log(0.9) / 1.0)
    tp = ls.photon_lifetime(0.3, 0.99, 0.95)
    assert tp == pytest.approx(0.6 / (ls.C_LIGHT * -np.log(0.99 * 0.95)))
    with pytest.raises(ValueError):
        ls.photon_lifetime(0.3, 1.0, 1.0)


def test_rigrod_optimum():
    g0l, L = 0.1, 0.02
    T = np.linspace(0.0, 0.2, 200001)
    p = ls.rigrod_output(T, g0l, L)
    assert T[np.argmax(p)] == pytest.approx(ls.optimal_output_coupling(g0l, L), abs=2e-6)
    assert ls.rigrod_output(0.5, g0l, L) == 0.0  # too much coupling: below threshold


def test_invalid_parameters():
    with pytest.raises(ValueError):
        ls.FourLevelLaser(tau=-1, tau_p=1, G=1)
    with pytest.raises(ValueError):
        ls.FourLevelLaser(tau=1, tau_p=1, G=1, beta=1.5)
