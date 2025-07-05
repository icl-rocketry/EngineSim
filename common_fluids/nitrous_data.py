from pyfluids import Fluid, Input, FluidsList
import matplotlib.pyplot as plt
import numpy as np

nitrous = Fluid(FluidsList.NitrousOxide)

T = np.linspace(-10, nitrous.critical_temperature, 1000)
P = np.zeros(len(T))
D_g = np.zeros(len(T))
D_l = np.zeros(len(T))

pressure = 36e5

for i, temp in enumerate(T):
    nitrous.update(Input.temperature(temp), Input.quality(0))
    P[i] = nitrous.pressure/1e5
    D_l[i] = nitrous.density
    nitrous.update(Input.temperature(temp), Input.quality(100))
    D_g[i] = nitrous.density

fig, ax1 = plt.subplots()

ax1.plot(T, P, 'r', label='Vapor Pressure')
ax1.grid()
ax1.grid(which="minor")
ax1.minorticks_on()
ax1.set_xlabel('Temperature (deg C)')
ax1.set_ylabel('Pressure (bar)', color='r')
ax1.set_title('Nitrous Properties vs Temperature')
ax1.set_xlim(left=T[0],right=T[-1])
ax2 = ax1.twinx()
ax2.plot(T, D_l,'b',label='Liquid Density')
# ax2.plot(T+273, D_g,'r',label='Gas Density')
# ax2.plot(T, D_g,'r',label='Gas Density')
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax2.set_ylabel('Density (kg/m^3)' , color='b')
ax2.minorticks_on()
# fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
fig.tight_layout()

t_amb = 10

nitrous.update(Input.temperature(t_amb), Input.quality(0))

print(f"Nitrous @ {t_amb:.1f} deg C -  saturation pressure: {nitrous.pressure/1e5:.2f} bar, density: {nitrous.density:.2f} kg/m^3")

nitrous.update(Input.pressure(28*1e5), Input.quality(0))
print(f"Nitrous vapor pressure: {nitrous.pressure/1e5:.2f} bar, density: {nitrous.density:.2f} kg/m^3")
nitrous.update(Input.pressure(32*1e5), Input.quality(0))
print(f"Nitrous vapor pressure: {nitrous.pressure/1e5:.2f} bar, density: {nitrous.density:.2f} kg/m^3")

plt.savefig('common_figs/nitrous_properties_vs_temp.png', dpi=300, bbox_inches='tight')
# plt.show()