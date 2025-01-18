import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

# Define the temperature range and pressure range
pressures = np.linspace(1e5, 60e5, 100)
substances = ['Methanol', 'Ethanol']
colors = ['b', 'g', 'r']

# Create the plot
plt.figure(figsize=(10, 6))

# Calculate and plot the boiling temperature for each pressure and each substance
for substance, color in zip(substances, colors):
    temperatures = []
    densities = []
    for pressure in pressures:
        density = 1000  # Initial density set above threshold to enter the loop
        temperature = 300  # Start at 300 K
        # Find the boiling temperature by iterating temperature until density drops to ~300 kg/m³ or below
        while density > 300 or temperature > 600:
            temperature += 0.1
            density = CP.PropsSI('D', 'T', temperature, 'P', pressure, substance)
        temperatures.append(temperature)
        densities.append(density)
    plt.plot(pressures / 1e5, temperatures, label=f'{substance}', color=color)

# Labeling the plot
plt.xlabel('Pressure (bar)')
plt.ylabel('Boiling Temperature (K)')
plt.title('Pressure vs Boiling Temperature for Methanol and Ethanol')
plt.legend()
plt.grid()
plt.show()