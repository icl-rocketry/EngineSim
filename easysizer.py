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

class RocketEngine:
    def __init__(self, oxName, fuelName, thrust, Pc, Pe, MR, radius_cylinder, length_chamber, cdo, cdf, Pi, density_o, density_f):
        self.oxName = oxName
        self.fuelName = fuelName
        self.Pc = Pc
        self.Pe = Pe
        self.thrust = thrust
        self.MR = MR
        self.radius_cylinder = radius_cylinder
        self.length_chamber = length_chamber
        self.cdo = cdo
        self.cdf = cdf
        self.Pi = Pi
        self.density_o = density_o
        self.density_f = density_f
    def runSim(self):
        # CEA
        self.cea = CEA_Obj(oxName = self.oxName,
                        fuelName = self.fuelName,
                        isp_units='sec',
                        cstar_units = 'm/s',
                        pressure_units='Bar',
                        temperature_units='K',
                        sonic_velocity_units='m/s',
                        enthalpy_units='J/g',
                        density_units='kg/m^3',
                        specific_heat_units='J/kg-K',
                        viscosity_units='lbf-sec/sqin',      # wtf
                        thermal_cond_units='mcal/cm-K-s',  # wtf
                        fac_CR=None,
                        make_debug_prints=False)

        self.eps = self.cea.get_eps_at_PcOvPe(Pc=self.Pc, MR=self.MR, PcOvPe = self.Pc/self.Pe)
        self.Tc, self.Tt, self.Te = self.cea.get_Temperatures(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.Pt = self.Pc / self.cea.get_Throat_PcOvPe(Pc=self.Pc, MR=self.MR)
        self.ve = self.cea.get_SonicVelocities(Pc=self.Pc,MR=self.MR,eps=self.eps)[2] * self.cea.get_MachNumber(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.mdot = self.thrust / self.ve
        self.mwt, self.gt = self.cea.get_Throat_MolWt_gamma(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.Rt = 8314.46 / self.mwt
        self.At = self.mdot / (self.Pt * 1e5) * np.sqrt(self.Rt * self.Tt / self.gt)
        self.CR = self.radius_cylinder**2 * np.pi / self.At
        self.cea = CEA_Obj(oxName = self.oxName,
                        fuelName = self.fuelName,
                        isp_units='sec',
                        cstar_units = 'm/s',
                        pressure_units='Bar',
                        temperature_units='K',
                        sonic_velocity_units='m/s',
                        enthalpy_units='J/g',
                        density_units='kg/m^3',
                        specific_heat_units='J/kg-K',
                        viscosity_units='lbf-sec/sqin',      # wtf
                        thermal_cond_units='mcal/cm-K-s',  # wtf
                        fac_CR=self.CR,
                        make_debug_prints=False)
        self.unitless_cea = rocketcea.cea_obj.CEA_Obj(oxName = self.oxName,fuelName = self.fuelName,fac_CR=self.CR,)

        self.eps = self.cea.get_eps_at_PcOvPe(Pc=self.Pc, MR=self.MR, PcOvPe = self.Pc/self.Pe)
        self.Tc, self.Tt, self.Te = self.cea.get_Temperatures(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.Pt = self.Pc / self.cea.get_Throat_PcOvPe(Pc=self.Pc, MR=self.MR)
        self.ve = self.cea.get_SonicVelocities(Pc=self.Pc,MR=self.MR,eps=self.eps)[2] * self.cea.get_MachNumber(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.mdot = self.thrust / self.ve

        self.mwt, self.gt = self.cea.get_Throat_MolWt_gamma(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.mwe, self.ge = self.cea.get_exit_MolWt_gamma(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.Rt = 8314.46 / self.mwt
        self.Re = 8314.46 / self.mwe

        self.Cp_c, self.Cp_t, self.Cp_e = self.cea.get_HeatCapacities(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.density_c = self.cea.get_Chamber_Density(Pc=self.Pc, MR=self.MR, eps=self.eps)

        _, self.oldvisc, self.oldk, self.oldpr = self.cea.get_Chamber_Transport(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.cstar = self.cea.get_Cstar(Pc=self.Pc, MR=self.MR)
        self.isp = self.cea.estimate_Ambient_Isp(Pc=self.Pc, MR=self.MR, eps=self.eps, Pamb=1)

        self.At = self.mdot / (self.Pt * 1e5) * np.sqrt(self.Rt * self.Tt / self.gt)
        self.Ae = self.At * self.eps
        self.Ac = self.radius_cylinder**2 * np.pi
        self.CR = self.Ac / self.At
        
        self.dt = np.sqrt(self.At / np.pi) * 2
        self.de = np.sqrt(self.Ae / np.pi) * 2
        self.lstar = self.length_chamber * self.radius_cylinder**2 / self.At
        
        self.fdot = self.mdot / (self.MR + 1)
        self.odot = self.mdot * self.MR / (self.MR + 1)
        
        self.Pd = self.Pi - self.Pc
        self.Ao = self.odot / self.cdo / np.sqrt(2 * self.density_o * self.Pd * 1e5);
        self.Af = self.fdot / self.cdf / np.sqrt(2 * self.density_f * self.Pd * 1e5);
        self.do = np.sqrt(self.Ao / np.pi) * 2
        self.df = np.sqrt(self.Af / np.pi) * 2
        
        
thanos = RocketEngine(
    oxName = "N2O",
    fuelName = "fuelmix",
    Pc = 20,
    Pe = 0.85,
    MR = 1.3, #thats O/F
    radius_cylinder=0.046,
    length_chamber=0.15,
    cdo = 0.4,
    cdf = 0.7,
    Pi = 30,
    density_o = 800,
    density_f = 792
)
fig, ax = plt.subplots(1, 2, figsize=(20, 7))
n_it = 10
waterlist = np.linspace(0, 20, n_it)
oxflux = np.zeros(n_it)
fuelflux = np.zeros(n_it)
duration = np.zeros(n_it)
for i, waterperc in enumerate(waterlist):
    fuelmix_str = (
    f'fuel CH3OH(L)   C 1 H 4 O 1 \n'
    f'h,cal=-57040.0      t(k)=298.15       wt%={100 - waterperc} \n'
    f'oxid water H 2 O 1  wt%={waterperc} \n'
    f'h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998 \n'
    )
    add_new_fuel( 'fuelmix', fuelmix_str )
    
    thanos.runSim()
    oxflux[i] = thanos.odot
    fuelflux[i] = thanos.fdot * 1.2
    
    coolantdensity = waterperc / 100 * 1000 + (100 - waterperc) / 100 * 800
    duration[i] = 8000 / coolantdensity / fuelflux[i]
ax[0].plot(waterlist, fuelflux)
ax[0].plot(waterlist, oxflux)
ax[1].plot(waterlist, duration)
    
# print(f'Mass flux (kg/s) = {thanos.mdot}   (g/s) = {thanos.mdot * 1000}')
# print()
# print(f'Fuel flux (kg/s) = {thanos.fdot}   (g/s) = {thanos.fdot * 1000}')
# print(f'Fuel Total(kg/s) = {thanos.fdot*1.2}   (g/s) = {thanos.fdot * 1000*1.2}')
# print(f'Fuel dia    (mm) = {thanos.df * 1000}')
# print()
# print(f'Ox flux   (kg/s) = {thanos.odot}   (g/s) = {thanos.odot * 1000}')
# print(f'Ox dia      (mm) = {thanos.do * 1000}')
# print()
# print(f'Throat Dia  (mm) = {thanos.dt * 1000}')
# print(f'Exit Dia    (mm) = {thanos.de * 1000}')
# print()
# print(f'ISP         (mm) = {thanos.isp}')
# print(f'CR          (mm) = {thanos.CR}')
# print(f'L*          (mm) = {thanos.lstar}')