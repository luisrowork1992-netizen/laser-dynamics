# laser-dynamics

![tests](https://github.com/luisrowork1992-netizen/laser-dynamics/actions/workflows/tests.yml/badge.svg)

`lasersim` simulates the **rate equations of a four-level laser**. It covers threshold behaviour, relaxation oscillations and Q-switched giant pulses, and adds closed-form formulas for designing a laser cavity.

## Features

- `FourLevelLaser(tau, tau_p, G, beta)`: inversion and photon-density rate equations
  - optional spontaneous-emission seeding β
- Exact steady state, with or without β, plus the threshold inversion and threshold pump rate
- Time-domain simulation with a constant or time-dependent pump
  - the photon density is integrated as ln S, so turn-on spikes that grow over 15 orders of magnitude stay stable
- Relaxation-oscillation frequency and damping
  - `measured_oscillation_frequency` extracts the frequency from a simulated trace
- Q-switching:
  - pulse simulation
  - analytic peak photon density, final inversion and energy-extraction efficiency
- Cavity formulas:
  - photon lifetime
  - threshold gain
  - longitudinal-mode spacing
  - Rigrod output power and the optimum output-coupler transmission

## Install

```bash
git clone https://github.com/luisrowork1992-netizen/laser-dynamics.git
cd laser-dynamics
pip install -e .
```

## Quick start

```python
import lasersim as ls

laser = ls.FourLevelLaser(tau=1.0, tau_p=1e-3, G=1.0)   # normalised units
print(laser.N_th, laser.P_th)                          # 1000, 1000
print(laser.steady_state(2 * laser.P_th))              # (1000.0, 1.0)  gain clamping

trace = laser.simulate(2 * laser.P_th, t_end=8.0, S0=1e-6)
print(laser.measured_oscillation_frequency(trace))     # ~31.5; theory sqrt((r-1)/(tau tau_p)) = 31.6

print(ls.qswitch_peak_photons(3000, 1000))             # 901.4
print(ls.mode_spacing(0.30) / 1e6, "MHz")              # 499.7
```

## Notebook

[`notebooks/laser_dynamics.ipynb`](notebooks/laser_dynamics.ipynb) covers five topics:

- the L–I curve and gain clamping
- turn-on spiking and relaxation oscillations
- a Q-switched pulse compared with theory
- Rigrod optimisation of the output coupler
- a worked example of a real 30 cm cavity

## Tests

```bash
pip install pytest
pytest
```

The tests check the following:

- simulations settle to the analytic steady state, below and above threshold
- the slope efficiency
- the steady state with β is an exact fixed point
- the measured relaxation-oscillation frequency matches √(ω_R² − γ²)
- Q-switched pulses at 1.5×, 3× and 6× threshold match the analytic peak photon density and final inversion, and the pulse peaks when N = N_th
- the extraction-efficiency equation
- the cavity formulas
- the Rigrod optimum agrees with a brute-force search

## Scope and limitations

- This is a single-mode, spatially uniform (point-model) laser. Spatial hole burning and multimode effects are not modelled.
- The ground-state population of a four-level system is taken as never depleted.

## References

- A. E. Siegman, *Lasers*, University Science Books (1986).
- O. Svelto, *Principles of Lasers*, 5th ed., Springer (2010).
- W. W. Rigrod, *J. Appl. Phys.* 36, 2487 (1965).

## License

MIT
