from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
import numpy as np
import rocketcea.cea_obj
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar
from pint import UnitRegistry
from pyfluids import Fluid, FluidsList, Input, Mixture
import pandas as pd
ureg = UnitRegistry()
Q_ = ureg.Quantity

pressures = np.linspace(1e5, 20e5, 100)
densities = np.zeros(100)
for i, pressure in enumerate(pressures:
    fuel = Fluid(FluidsList.Methanol).with_state(Input.pressure(pressure), Input.temperature(5))
    densities[i] = fuel.density
plt.plot(pressures, densities)