from os import system
from rocketcea.cea_obj_w_units import CEA_Obj
from pyfluids import Fluid, FluidsList, Input
from thermo import Chemical
import numpy as np

cea = CEA_Obj(oxName="N2O", fuelName="Isopropanol",
                isp_units='sec',
                cstar_units = 'm/s',
                pressure_units='Bar',
                temperature_units='K',
                sonic_velocity_units='m/s',
                enthalpy_units='J/g',
                density_units='kg/m^3',
                specific_heat_units='J/kg-K',
                viscosity_units='centipoise', # stored value in pa-s
                thermal_cond_units='W/cm-degC', # stored value in W/m-K
                make_debug_prints=False, fac_CR=4.7059)

# A_t = 1412.62 * 1e-6  # m^2, throat area
A_t = 1400 * 1e-6
A_e = 5310.68 * 1e-6  # m^2, exit area
eps = A_e / A_t  # expansion ratio
print(eps)

# Pintle
fuel_core_cd_estimated = 0.8
fuel_core_area = 21.106 * 1e-6
# fuel_core_CdA = fuel_core_area * fuel_core_cd_estimated
fuel_film_area = 5.640 * 1e-6
film_cd = 0.65
# fuel_film_CdA = fuel_film_area * film_cd

ox_cd_estimated = 0.4
ox_area = 106.029 * 1e-6
# ox_CdA = ox_area * ox_cd_estimated    

# ------------------------
# TEST DATA

pc = 14.34
thrust = 2700
p_f = 27.63 
p_ox = 24.5
p_feedf = 32.26
ox_mdot = 1.199
fuel_mdot = 0.648

pc = 17.31
thrust = 3160
p_f = 28.01
p_ox = 26.43
p_feedf = 32.26
ox_mdot = 1.359
fuel_mdot = 0.599

pc = 16.53
thrust = 3000
p_f = 28.08
p_ox = 25.0
p_feedf = 32.12
#nitrous temp 1.5c
ox_mdot = 1.302
fuel_mdot = 0.621

pc = 25.4
thrust = 4800
p_f = 44.29
p_ox = 33.01
p_feedf = 50.43
ox_mdot = 1.958
fuel_mdot = 0.771

pc = 28.93
thrust = 5500
p_f = 50.3
p_ox = 33.97
p_feedf = 57.41
ox_mdot = 2.193
fuel_mdot = 0.83

pc = 29.5
thrust = 5583.4
p_f = 43.1
p_ox = 34.27
p_feedf = 49.25
ox_mdot = 2.228
fuel_mdot = 0.764

pc = 30.79
thrust = 5803 + 120
p_f = 43
p_ox = 35.7
p_feedf = 49.2 
ox_mdot = 2.333
fuel_mdot = 0.761

gauge = True
if gauge == True:
    pc += 1
    p_f += 1
    p_ox += 1
    p_feedf += 1

# ------------------------
# DESIRED SETPOINT
set_pc = 29.3
set_OF = 2.9
# ------------------------
# Coolprop doesn't have isopropanol, so we use Thermo
# ipa = Chemical('Isopropanol', T=298.15, P=p_f * 1e5)
# n2o = Fluid(FluidsList.NitrousOxide)
# n2o.update(Input.pressure(p_ox * 1e5), Input.quality(0))
# ipa_density = ipa.rho
# n2o_density = n2o.density

fuel_density = 790
fuel_density = 675
ox_density = 898.907


dp_f = p_f - pc
stiff_f = dp_f / pc
fuel_cda = fuel_mdot / np.sqrt(2 * fuel_density * dp_f * 1e5)

dp_ox = p_ox - pc
stiff_ox = dp_ox / pc
ox_cda = ox_mdot / np.sqrt(2 * ox_density * dp_ox * 1e5)


OF = ox_mdot / fuel_mdot
mdot = ox_mdot + fuel_mdot
fuel_film_mdot = film_cd * np.sqrt(2 * fuel_density * dp_f * 1e5) * fuel_film_area
fuel_core_mdot = fuel_mdot - fuel_film_mdot
# fuel_core_mdot = fuel_mdot * fuel_core_CdA / (fuel_core_CdA + fuel_film_CdA)
# fuel_film_mdot = fuel_mdot * fuel_film_CdA / (fuel_core_CdA + fuel_film_CdA)

core_OF = ox_mdot / fuel_core_mdot
core_mdot = ox_mdot + fuel_core_mdot

isp_core = cea.estimate_Ambient_Isp(pc, MR=core_OF, eps=eps, Pamb=1.013)[0]
# isp_theory = isp_core * core_mdot / mdot
isp_theory = isp_core * (fuel_mdot * 0.8 + ox_mdot) / mdot
isp_actual = thrust / (mdot * 9.81)


cstar_theory = cea.get_Cstar(pc, MR=OF);
cstar_actual = pc * 1e5 * A_t / mdot;

cf_theory = cea.get_PambCf(Pamb=1.013, Pc=pc, MR=OF, eps=eps)[0]
cf_actual = thrust / (pc * 1e5 * A_t)

cd_f_actual = fuel_core_mdot / (fuel_core_area * np.sqrt(2 * fuel_density * dp_f * 1e5))
cda_f_actual = fuel_mdot / (np.sqrt(2 * fuel_density * dp_f * 1e5))
cd_ox_actual = ox_mdot / (ox_area * np.sqrt(2 * ox_density * dp_ox * 1e5))

dp_regen = p_feedf - p_f
regen_cda = fuel_mdot / np.sqrt(2 * fuel_density * dp_regen * 1e5)

# Desired calculation
set_ve = cea.get_SonicVelocities(Pc=set_pc,MR=set_OF,eps=eps)[2] * cea.get_MachNumber(Pc=set_pc, MR=set_OF, eps=eps)
set_thrust = cf_actual * set_pc * 1e5 * A_t
# set_mdot = set_thrust / set_ve / (cstar_actual / cstar_theory)
set_mdot = set_thrust / thrust * mdot
set_ox_mdot = set_OF * set_mdot / (1 + set_OF)
set_fuel_mdot = set_mdot - set_ox_mdot
# set_fuel_core_mdot = set_fuel_mdot * fuel_core_area * cd_f_actual / (fuel_core_area * cd_f_actual + fuel_film_area * film_cd)
# set_core_of = set_ox_mdot / set_fuel_core_mdot
set_p_ox = set_pc + set_ox_mdot**2 / (cd_ox_actual**2 * ox_area**2 * 2 * ox_density) / 1e5
set_p_f = set_pc + set_fuel_mdot**2 / (cda_f_actual**2 * 2 * fuel_density) / 1e5 + set_fuel_mdot**2 / (regen_cda**2 * 2 * fuel_density) / 1e5

if __name__ == "__main__":
    # system('cls')
    print('-'* 50)
    print(f'{"Pc":<20} {pc:<10.2f} Bar')
    print(f'{"Fuel Mdot":<20} {fuel_mdot:<10.4f} kg/s')
    print(f'{"Ox Mdot":<20} {ox_mdot:<10.4f} kg/s')
    print(f'{"Thrust":<20} {thrust:<10.2f} N')
    print(f'{"Fuel Pressure":<20} {p_f:<10.2f} Bar')
    print(f'{"Ox Pressure":<20} {p_ox:<10.2f} Bar')
    print(f'{"Fuel Feed Pressure":<20} {p_feedf:<10.2f} Bar')
    print("")
    print(f'{"Mdot":<20} {mdot:<10.4f} kg/s')
    print(f'{"Mdot Core":<20} {core_mdot:<10.4f} kg/s')
    print(f'{"O/F":<20} {OF:<10.4f}')
    print(f'{"O/F Core":<20} {core_OF:<10.4f}')
    print(f'{"Fuel Core Mdot":<20} {fuel_core_mdot:<10.4f} kg/s')
    print(f'{"Fuel Film Mdot":<20} {fuel_film_mdot:<10.4f} kg/s')

    # print(f'{"Fuel density":<20} {ipa.rho:<10.2f} kg/m^3')
    # print(f'{"Ox density":<20} {n2o.density:<10.2f} kg/m^3')
    # print(f'{"N2O vapor pressure":<20} {n2o.pressure/1e5:<10.2f} Bar')
    # print(f'{"N2O saturation temperature":<20} {n2o.temperature:<10.2f} K')
    print("")
    print(f'{"ISP Theory":<20} {isp_theory:<10.2f} sec')
    print(f'{"ISP Actual":<20} {isp_actual:<10.2f} sec')
    print(f'{"ISP Efficiency":<20} {isp_actual/isp_theory:<10.4f}')
    print("")
    print(f'{"C* Theory":<20} {cstar_theory:<10.2f} m/s')
    print(f'{"C* Actual":<20} {cstar_actual:<10.2f} m/s')
    print(f'{"C* Efficiency":<20} {cstar_actual/cstar_theory:<10.4f}')
    print("")
    print(f'{"CF Theory (Dubious)":<20} {cf_theory:<10.4f}')
    print(f'{"CF Actual":<20} {cf_actual:<10.4f}')
    print(f'{"CF Efficiency":<20} {cf_actual/cf_theory:<10.4f}')
    print("")
    print(f'{"Fuel DP":<20} {dp_f:<10.2f} Bar')
    print(f'{"Fuel Stiffness":<20} {stiff_f:<10.4f}')
    print(f'{"Fuel Core Cd":<20} {cd_f_actual:<10.4f}')
    print(f'{"Fuel CdA":<20} {cda_f_actual*1e6:<10.4f} mm^2')
    print("")
    print(f'{"Ox DP":<20} {dp_ox:<10.2f} Bar')
    print(f'{"Ox Stiffness":<20} {stiff_ox:<10.4f}')
    print(f'{"Ox Cd":<20} {cd_ox_actual:<10.4f}')
    print("")
    print(f'{"Regen DP":<20} {dp_regen:<10.2f} Bar')
    print(f'{"Regen CdA":<20} {regen_cda*1e6:<10.4f} mm^2')
    print('-'* 50)
    print(f'{"Set Pc":<20} {set_pc:<10.2f} Bar')
    print(f'{"Set O/F":<20} {set_OF:<10.4f}')
    # print(f'{"Set OF Core":<20} {set_core_of:<10.4f}')
    print(f'{"Set Thrust":<20} {set_thrust:<10.2f} N')
    print(f'{"Set Mdot":<20} {set_mdot:<10.4f} kg/s')
    print(f'{"Set Fuel Mdot":<20} {set_fuel_mdot:<10.4f} kg/s')
    print(f'{"Set Ox Mdot":<20} {set_ox_mdot:<10.4f} kg/s')
    print(f'{"Set Ox Pressure":<20} {set_p_ox:<10.2f} Bar')
    print(f'{"Set Fuel Pressure":<20} {set_p_f:<10.2f} Bar')
    print('-'* 50)