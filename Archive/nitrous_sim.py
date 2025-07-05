import matplotlib.pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI

# Define the temperature range in Celsius and convert to Kelvin
temp_celsius = np.linspace(-30, 36.37, 500)
temp_kelvin = temp_celsius + 273.15  # Convert to Kelvin

# Initialize lists to store the properties
pressure_vapor = []  # Saturated vapor pressure
density_liquid = []  # Saturated liquid density
density_vapor = []   # Saturated vapor density

# Calculate properties for each temperature
for T in temp_kelvin:
    pressure_vapor.append(PropsSI('P', 'T', T, 'Q', 1, 'NitrousOxide') / 1e5)  # Convert pressure to bar
    density_liquid.append(PropsSI('D', 'T', T, 'Q', 0, 'NitrousOxide'))       # Liquid density in kg/m³
    density_vapor.append(PropsSI('D', 'T', T, 'Q', 1, 'NitrousOxide'))        # Vapor density in kg/m³

# Create the plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot pressures on the first y-axis
ax1.plot(temp_celsius, pressure_vapor, label="Saturated Vapor Pressure (bar)", color="red")
ax1.set_xlabel("Temperature (°C)")
ax1.set_ylabel("Pressure (bar)", color="red")
ax1.tick_params(axis='y', labelcolor="red")
plt.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
plt.minorticks_on()

# Create a second y-axis for densities
ax2 = ax1.twinx()
ax2.plot(temp_celsius, density_liquid, label="Saturated Liquid Density (kg/m³)", color="blue")
ax2.plot(temp_celsius, density_vapor, label="Saturated Vapor Density (kg/m³)", color="green")
ax2.set_ylabel("Density (kg/m³)", color="blue")
ax2.tick_params(axis='y', labelcolor="blue")

# Add legends for both axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left")

plt.title("Properties of Nitrous Oxide vs. Temperature")
plt.show()