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
    fuelName = "Methanol",
    thrust = 1500,
    Pc = 20,
    Pe = 1,
    MR = 2.5, #thats O/F
    radius_cylinder=0.046,
    length_chamber=0.15,
    cdo = 0.4,
    cdf = 0.7,
    Pi = 30,
    density_o = 800,
    density_f = 792
)
#thanos.runSim()

n_1 = 5

def newround(x):
    return round(x, -int(np.floor(np.log10(abs(x)))) + 4)

fig, ax = plt.subplots(2, 1, figsize=(7, 10))
MRlist = np.linspace(0.5, 5, n_1)
pclist = np.linspace(20, 40, 5)
mdotlist = np.zeros(n_1)
isplist = np.zeros(n_1)
for j, pc in enumerate(pclist):
    for i, MR in enumerate(MRlist):
        thanos.MR = MR
        thanos.Pc = pc
        thanos.runSim()
        cstarlist[i] = thanos.cstar
        cflist[i] = thanos.thrust / (thanos.Pc * 1e5 * thanos.At)
        templist[i] = thanos.Tc
        isplist[i] = thanos.isp[0]
    ax[0].plot(MRlist, templist,color=[0.2 + pc / 50, 0.2, 0.5] , label=f'PC:{pc}')
    ax[0].set_xlabel("O/F")
    ax[0].set_ylabel("Chamber Temperature (K)")
    ax[1].plot(MRlist, isplist,color=[0.2 + pc / 50, 0.2, 0.5] , label=f'PC%:{pc}')
    ax[1].set_xlabel("O/F")
    ax[1].set_ylabel("ISP (s)")
ax[0].legend()
ax[1].legend()
ax[0].grid()
ax[1].grid()
ax[0].minorticks_on()
ax[1].minorticks_on()
plt.style.use('seaborn-v0_8-dark')
plt.show()