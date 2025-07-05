import numpy as np
import scipy as sp
from pyfluids import Input

def spi_model(P_up, P_down, rho_up, A, Cd):
    """Single-Phase Incompressible flow model"""
    delta_p = P_up - P_down
    return Cd * A * np.sqrt(2 * rho_up * delta_p)

def hem_model(upstream, downstream_p, A, Cd):
    """Homogeneous Equilibrium Model"""

    def HEMfunc(upstream, downstream_p, A, Cd):
        downstream = upstream.clone()
        downstream.update(Input.pressure(downstream_p), Input.entropy(upstream.entropy))
        return Cd * A * downstream.density * np.sqrt(2 * (upstream.enthalpy - downstream.enthalpy)) if upstream.enthalpy > downstream.enthalpy else 0

    sol = sp.optimize.minimize_scalar(lambda downstream_p: -HEMfunc(upstream, downstream_p, A, Cd), bounds=[9e4,upstream.pressure], method='bounded')

    choked_p = sol.x
    choked_mdot = -sol.fun

    if (choked_p > downstream_p):
        mdot = choked_mdot
    else:
        mdot = HEMfunc(upstream, downstream_p, A, Cd)

    return mdot

def nhne_model(mdot_spi, mdot_hem, k):
    """Non-Homogeneous Non-Equilibrium Model"""
    return (mdot_spi * k / (1 + k)) + (mdot_hem / (1 + k))