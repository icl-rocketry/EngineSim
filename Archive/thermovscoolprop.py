import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from thermo.chemical import Chemical

# Define temperature range (in °C)
temps_C = np.linspace(-50, 30, 100)  # Temperature in Celsius
temps_K = temps_C + 273.15  # Convert to Kelvin

# Calculate density using CoolProp
coolprop_densities = []
for T in temps_K:
    try:
        density = PropsSI("P", "T", T, "Q", 1, "NitrousOxide")
        coolprop_densities.append(density/1e5)
    except ValueError:
        coolprop_densities.append(None)  # Handle invalid values

# Calculate density using Thermo
thermo_densities = []
ethanol = Chemical("N2O")  # Create ethanol chemical object
for T in temps_C:
    ethanol.T = T + 273.15  # Set temperature (Thermo uses Kelvin internally)
    thermo_densities.append(ethanol.Psat/1e5)

# Filter out invalid CoolProp data (if any)
valid_temps_C = [T for T, d in zip(temps_C, coolprop_densities) if d is not None]
valid_coolprop_densities = [d for d in coolprop_densities if d is not None]

# Plot the results
plt.figure(figsize=(10, 6))

# CoolProp data
plt.plot(valid_temps_C, valid_coolprop_densities, label="CoolProp Vapor Pressure", color="blue")

# Thermo data
plt.plot(temps_C, thermo_densities, label="Thermo Vapor Pressure", linestyle="--", color="orange")

# Plot settings
plt.xlabel("Temperature (°C)")
plt.ylabel("Vapor Pressure (bar)")
plt.title("Vapor Pressure of Nitrous Oxide vs Temperature (-50°C to 30°C)")
plt.grid(True)
plt.legend()
plt.show()