from rocketcea.cea_obj_w_units import CEA_Obj
import numpy as np
from matplotlib import pyplot as plt
Pc=10
MR=2.5
Pe=0.8
thrust=4000
radius_cylinder=0.04

ispArray = []
CRObj = []
CRObj.append(CEA_Obj(oxName = "N2O",
                    fuelName = "Methanol",
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
                    make_debug_prints=False))
for i in range(1,10):
    cea = CRObj[i-1]
    eps = cea.get_eps_at_PcOvPe(Pc=Pc, MR=MR, PcOvPe = Pc/Pe)
    a, b = cea.estimate_Ambient_Isp(Pc=Pc, MR=MR, eps=eps, Pamb=1)
    ispArray.append(a)
    Tc, Tt, Te = cea.get_Temperatures(Pc=Pc, MR=MR, eps=eps)
    Pt = Pc / cea.get_Throat_PcOvPe(Pc=Pc, MR=MR)
    ve = cea.get_SonicVelocities(Pc=Pc,MR=MR,eps=eps)[2] * cea.get_MachNumber(Pc=Pc, MR=MR, eps=eps)
    mdot = thrust / ve
    mwt, gt = cea.get_Throat_MolWt_gamma(Pc=Pc, MR=MR, eps=eps)
    Rt = 8314.46 / mwt
    At = mdot / (Pt * 1e5) * np.sqrt(Rt * Tt / gt)
    CR = radius_cylinder**2 * np.pi / At
    CRObj.append(CEA_Obj(oxName = "N2O",
                        fuelName = "Methanol",
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
                        fac_CR=CR,
                        make_debug_prints=False))
cea = CRObj[9]
eps = cea.get_eps_at_PcOvPe(Pc=Pc, MR=MR, PcOvPe = Pc/Pe)
a, b = cea.estimate_Ambient_Isp(Pc=Pc, MR=MR, eps=eps, Pamb=1)
ispArray.append(a)
print(ispArray)
[plt.plot([0,1,2,3,4,5,6,7,8,9],ispArray)]