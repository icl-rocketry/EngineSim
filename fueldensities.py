import numpy as np
import matplotlib.pyplot as plt
from thermo.chemical import Chemical

# Define chemicals
chemicals = {
    "Methanol": Chemical("methanol"),
    "Ethanol": Chemical("ethanol"),
    "Isopropanol": Chemical("isopropanol"),
}

# Define temperature range (in Celsius)
temps_C = np.linspace(-100, 200, 100)  # Temperature range in °C
temps_K = temps_C + 273.15  # Convert to Kelvin

# Prepare data storage
properties = {chemical: {"Density": [], "Thermal Conductivity": [], "Heat Capacity": []} for chemical in chemicals}

# Calculate properties for each chemical over the temperature range
for name, chem in chemicals.items():
    for T in temps_K:
        chem.T = T
        properties[name]["Density"].append(chem.rho)
        properties[name]["Thermal Conductivity"].append(chem.k)
        properties[name]["Heat Capacity"].append(chem.Cp)

# Create subplots
fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

# Plot Density
for name, data in properties.items():
    axes[0].plot(temps_C, data["Density"], label=name)
axes[0].set_title("Density vs Temperature")
axes[0].set_ylabel("Density (kg/m³)")
axes[0].grid(True)
axes[0].legend()

# Plot Thermal Conductivity
for name, data in properties.items():
    axes[1].plot(temps_C, data["Thermal Conductivity"], label=name)
axes[1].set_title("Thermal Conductivity vs Temperature")
axes[1].set_ylabel("Thermal Conductivity (W/m·K)")
axes[1].grid(True)
axes[1].legend()

# Plot Heat Capacity
for name, data in properties.items():
    axes[2].plot(temps_C, data["Heat Capacity"], label=name)
axes[2].set_title("Heat Capacity vs Temperature")
axes[2].set_xlabel("Temperature (°C)")
axes[2].set_ylabel("Heat Capacity (J/mol·K)")
axes[2].grid(True)
axes[2].legend()

# Show plot
plt.tight_layout()
plt.show()