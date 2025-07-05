import numpy as np
from pyfluids import Fluid, FluidsList, Input
import matplotlib.pyplot as plt
from os import system
from flow_models import *

system("cls")

tank_p = 40e5
tank_vapor_p = 28e5
injector_p = 30e5
chamber_p = 15e5

Cd = 0.65

pintle_d = 17.27e-3
pintle_id = 12e-3
hole_d = 1.5e-3
n_holes_row = 12
n_rows = 5

inlet_area = np.pi * (pintle_id / 2) ** 2
pintle_circumference = np.pi * pintle_d
blockage_ratio = (n_holes_row * hole_d) / pintle_circumference
A_inj =  n_rows * n_holes_row * np.pi * (hole_d / 2) ** 2
# A_inj = 118.609 * 0.75 * 1e-6
A_nhne = A_inj/ 1.1

nitrous = Fluid(FluidsList.NitrousOxide) # Random nitrous instance to get properties
nitrous_tank = Fluid(FluidsList.NitrousOxide)
nitrous_injector = Fluid(FluidsList.NitrousOxide)
nitrous_chamber = Fluid(FluidsList.NitrousOxide)

# Setup nitrous tank state, modelling filling as saturated liquid and then pressurisation (supercharging)
nitrous_tank.update(Input.pressure(tank_vapor_p), Input.quality(0))  # Saturated liquid
fill_temp = nitrous_tank.temperature
nitrous_tank.update(Input.entropy(nitrous_tank.entropy), Input.pressure(tank_p)) # Supercharged liquid from saturation

# Setup nitrous injector state
nitrous_injector.update(Input.pressure(injector_p), Input.enthalpy(nitrous_tank.enthalpy))  # Isenthalpic expansion to below saturation (two phase) e.g. dp across feed system + valve
injector_T = nitrous_injector.temperature
nitrous.update(Input.temperature(injector_T), Input.quality(0))
injector_vapor_p = nitrous.pressure
injector_sat_density = nitrous.density

# Calculate mass flow rates
mdot_spi = spi_model(injector_p, chamber_p, nitrous_injector.density, A_nhne, Cd)
mdot_hem = hem_model(nitrous_injector, chamber_p, A_nhne, Cd)
k = np.sqrt((injector_p - chamber_p) / (injector_vapor_p - chamber_p))

mdot_nhne = nhne_model(mdot_spi, mdot_hem, k)

print(f"Tank: p: {nitrous_tank.pressure/1e5:.2f} Bar, vp: {tank_vapor_p/1e5:.2f} Bar, T: {nitrous_tank.temperature:.2f} deg C, fill T: {fill_temp:.2f} deg C, phase: {nitrous_tank.phase}")

print(f"\nInjector: p: {nitrous_injector.pressure/1e5:.2f} Bar, vp: {injector_vapor_p/1e5:.2f} Bar, temperature: {injector_T:.2f} deg C")# , quality: {nitrous_injector.quality:.2f} %, ")
print(f"          density: {nitrous_injector.density:.2f} kg/m^3, saturated density: {injector_sat_density:.2f} kg/m^3, phase: {nitrous_injector.phase}")

print(f"\nNHNE mdot: {mdot_nhne:.4f} kg/s, SPI mdot: {mdot_spi:.4f} kg/s, HEM mdot: {mdot_hem:.4f} kg/s")

Cd_eff = mdot_nhne / (A_inj * np.sqrt(2 * nitrous_injector.density * (injector_p - chamber_p)))
Cd_ratio = Cd_eff / Cd
print(f"Effective Cd: {Cd_eff:.4f}, Cd ratio: {Cd_ratio:.4f}")

N = 100
injector_p_arr = np.linspace(chamber_p+1, tank_p-1e4, 100)
chamber_p_arr = np.zeros(N)
mdot_spi_arr = np.zeros(N)
mdot_hem_arr = np.zeros(N)
mdot_nhne_arr = np.zeros(N)
k_arr = np.zeros(N)
injector_vapor_p_arr = np.zeros(N)
injector_density_arr = np.zeros(N)
injector_sat_density_arr_L = np.zeros(N)
injector_sat_density_arr_V = np.zeros(N)
injector_vapor_quality_arr = np.zeros(N)
injector_T_arr = np.zeros(N)

p_arr = injector_p_arr

for i, inj_p in enumerate(injector_p_arr):
    nitrous_tank.update(Input.pressure(tank_vapor_p), Input.quality(0))  # Saturated liquid
    fill_temp = nitrous_tank.temperature
    nitrous_tank.update(Input.entropy(nitrous_tank.entropy), Input.pressure(tank_p)) # Supercharged liquid from saturation

    nitrous_injector.update(Input.pressure(inj_p), Input.entropy(nitrous_tank.entropy))  # Isentropic expansion to below saturation (two phase)
    injector_T_arr[i] = nitrous_injector.temperature
    injector_density_arr[i] = nitrous_injector.density
    injector_vapor_quality_arr[i] = nitrous_injector.quality if nitrous_injector.quality is not None else 0
    nitrous.update(Input.temperature(injector_T_arr[i]), Input.quality(0))
    injector_vapor_p_arr[i] = nitrous.pressure
    injector_sat_density_arr_L[i] = nitrous.density
    nitrous.update(Input.temperature(injector_T_arr[i]), Input.quality(100))
    injector_sat_density_arr_V[i] = nitrous.density

    mdot_spi_arr[i] = spi_model(inj_p, chamber_p, nitrous_injector.density, A_nhne, Cd)
    mdot_hem_arr[i] = hem_model(nitrous_injector, chamber_p, A_nhne, Cd)
    k_arr[i] = np.sqrt((inj_p - chamber_p) / (injector_vapor_p_arr[i] - chamber_p))
    mdot_nhne_arr[i] = nhne_model(mdot_spi_arr[i], mdot_hem_arr[i], k_arr[i])


# Plotting the results in subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Two-Phase Injector Model Analysis', fontsize=16)

# Unpack axes for easier naming
ax_mdot, ax_k, ax_vp = axes[0]
ax_temp, ax_density, ax_quality = axes[1]

# Mass flow rate plot
ax_mdot.plot(p_arr / 1e5, mdot_spi_arr, label='SPI Model', color='blue')
ax_mdot.plot(p_arr / 1e5, mdot_hem_arr, label='HEM Model', color='orange')
ax_mdot.plot(p_arr / 1e5, mdot_nhne_arr, label='NHNE Model', color='green')
ax_mdot.axvline(x=tank_vapor_p / 1e5, color='black', linestyle='--', alpha=0.7, label='Tank Vapor Pressure')
ax_mdot.set_ylim(0, None)
ax_mdot.set_xlim(p_arr[0] / 1e5, p_arr[-1] / 1e5)
ax_mdot.set_title('Mass Flow Rate vs Injector Pressure')
ax_mdot.set_xlabel('Injector Pressure (Bar)')
ax_mdot.set_ylabel('Mass Flow Rate (kg/s)')
ax_mdot.legend()
ax_mdot.grid()

# k factor plot
ax_k.plot(p_arr / 1e5, k_arr, label='k Factor', color='blue')
ax_k.axvline(x=tank_vapor_p / 1e5, color='black', linestyle='--', alpha=0.7, label='Tank Vapor Pressure')
ax_k.set_title('k Factor vs Injector Pressure')
ax_k.set_xlabel('Injector Pressure (Bar)')
ax_k.set_ylabel('k Factor')
ax_k.legend()
ax_k.grid()
ax_k.set_ylim(0, None)
ax_k.set_xlim(p_arr[0] / 1e5, p_arr[-1] / 1e5)

# Vapor pressure plot
ax_vp.plot(p_arr / 1e5, injector_vapor_p_arr / 1e5, label='Injector Vapor Pressure', color='blue')
ax_vp.axvline(x=tank_vapor_p / 1e5, color='black', linestyle='--', alpha=0.7, label='Tank Vapor Pressure')
ax_vp.set_title('Injector Vapor Pressure vs Injector Pressure')
ax_vp.set_xlabel('Injector Pressure (Bar)')
ax_vp.set_ylabel('Injector Vapor Pressure (Bar)')
ax_vp.legend()
ax_vp.grid()
ax_vp.set_ylim(0, None)
ax_vp.set_xlim(p_arr[0] / 1e5, p_arr[-1] / 1e5)

# Temperature plot
ax_temp.plot(p_arr / 1e5, injector_T_arr, label='Injector Temperature', color='blue')
ax_temp.axhline(y=nitrous_tank.temperature, color='green', linestyle='--', alpha=0.7, label='Tank Temperature')
ax_temp.axhline(y=fill_temp, color='red', linestyle='--', alpha=0.7, label='Fill Temperature')
ax_temp.axvline(x=tank_vapor_p / 1e5, color='black', linestyle='--', alpha=0.7, label='Tank Vapor Pressure')
ax_temp.set_title('Injector Temperature vs Injector Pressure')
ax_temp.set_xlabel('Injector Pressure (Bar)')
ax_temp.set_ylabel('Injector Temperature (C)')
ax_temp.legend()
ax_temp.grid()
ax_temp.set_xlim(p_arr[0] / 1e5, p_arr[-1] / 1e5)

# Density plot
ax_density.plot(p_arr / 1e5, injector_sat_density_arr_L, label='Injector Saturated Liquid Density', color='red')
ax_density.plot(p_arr / 1e5, injector_sat_density_arr_V, label='Injector Saturated Vapor Density', color='orange')
ax_density.plot(p_arr / 1e5, injector_density_arr, label='Injector Density', color='blue')
ax_density.axhline(y=nitrous_tank.density, color='green', linestyle='--', alpha=0.7, label='Tank Density')
ax_density.axvline(x=tank_vapor_p / 1e5, color='black', linestyle='--', alpha=0.7, label='Tank Vapor Pressure')
ax_density.set_title('Injector Density vs Injector Pressure')
ax_density.set_xlabel('Injector Pressure (Bar)')
ax_density.set_ylabel('Injector Density (kg/m^3)')
ax_density.legend()
ax_density.grid()
ax_density.set_ylim(0, None)
ax_density.set_xlim(p_arr[0] / 1e5, p_arr[-1] / 1e5)

# Vapor quality plot
ax_quality.plot(p_arr / 1e5, injector_vapor_quality_arr, label='Injector Vapor Quality', color='blue')
ax_quality.axvline(x=tank_vapor_p / 1e5, color='black', linestyle='--', alpha=0.7, label='Tank Vapor Pressure')
ax_quality.set_title('Injector Vapor Quality vs Injector Pressure')
ax_quality.set_xlabel('Injector Pressure (Bar)')
ax_quality.set_ylabel('Vapor Quality')
ax_quality.legend()
ax_quality.grid()
ax_quality.set_ylim(0, None)
ax_quality.set_xlim(p_arr[0] / 1e5, p_arr[-1] / 1e5)

plt.tight_layout()
plt.show()