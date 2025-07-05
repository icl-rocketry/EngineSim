import numpy as np
from thermo import Chemical

import matplotlib.pyplot as plt

# Create a Chemical object for isopropyl alcohol (IPA)
# ipa = Chemical('isopropanol')  # Can also use '2-propanol' or CAS '67-63-0'

# Define temperature range in Kelvin (20°C to 100°C)
temp_range_K = np.linspace(293.15, 373.15+100, 100)
# Calculate vapor pressure for each temperature
vapor_pressures_Pa = [Chemical('isopropanol', T=T).Psat for T in temp_range_K]
vapor_pressures_bar = [p/1e5 for p in vapor_pressures_Pa]  # Convert to bar

# Calculate liquid density for each temperature (kg/m³)
densities_kg_m3 = [Chemical('isopropanol', T=T).rhol for T in temp_range_K]

# Convert temperatures to Celsius for plotting
temp_range_C = temp_range_K - 273.15

# Create the plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(12, 8))

# Plot vapor pressure on the left y-axis
color = 'tab:blue'
ax1.set_xlabel('Temperature (°C)')
ax1.set_ylabel('Vapor Pressure (bar)', color=color)
ax1.plot(temp_range_C, vapor_pressures_bar, 'b-', linewidth=2, label='Vapor Pressure')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, which="both", ls="--", alpha=0.3)

# Create second y-axis for density
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Liquid Density (kg/m³)', color=color)
ax2.plot(temp_range_C, densities_kg_m3, 'r-', linewidth=2, label='Liquid Density')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Isopropyl Alcohol (IPA) Properties vs Temperature')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
plt.savefig('common_figs/ipa_properties_vs_temp.png', dpi=300)
# plt.show()