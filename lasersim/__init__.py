"""lasersim: four-level laser rate equations and laser-cavity design formulas."""

from .analytic import (
    C_LIGHT,
    mode_spacing,
    optimal_output_coupling,
    photon_lifetime,
    qswitch_extraction_efficiency,
    qswitch_final_inversion,
    qswitch_peak_photons,
    rigrod_output,
    threshold_gain,
)
from .rate_equations import FourLevelLaser, Trace

__version__ = "0.1.0"
