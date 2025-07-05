import numpy as np
import matplotlib.pyplot as plt
from thermo.chemical import Chemical
from scipy.optimize import brentq  # For numerical root finding

# Create chemical objects for methanol, ethanol, and isopropanol
methanol = Chemical("methanol")
ethanol = Chemical("ethanol")
isopropanol = Chemical("isopropanol")

# List of chemicals for ease of computation
chemicals = {"Methanol": methanol, "Ethanol": ethanol, "Isopropanol": isopropanol}

# Define pressure range from 1 bar to 50 bar (converted to Pascals)
pressures_bar = np.linspace(1, 50, 100)  # Pressures in bar
pressures_pa = pressures_bar * 1e5  # Convert bar to Pascals

# Define a function to compute the boiling point at a given pressure
def find_boiling_point(chem, P_target):
    """
    Function to numerically find the boiling point (saturation temperature) 
    at a given target pressure for a specified chemical.
    """
    def func(T):
        chem.T = T  # Set the temperature
        return chem.Psat - P_target  # Difference between Psat and target pressure

    # Solve for T such that Psat(T) == P_target using brentq
    T_sat = brentq(func, 200, 600)  # Search within a reasonable temperature range (in Kelvin)
    return T_sat


# Calculate boiling points over the pressure range for each chemical
boiling_points = {name: [] for name in chemicals}

for name, chem in chemicals.items():
    for P in pressures_pa:
        try:
            T_sat = find_boiling_point(chem, P)  # Find boiling point at given pressure
            boiling_points[name].append(T_sat - 273.15)  # Convert from Kelvin to Celsius
        except ValueError:
            boiling_points[name].append(None)  # Handle invalid states

# Plotting
plt.figure(figsize=(12, 8))

for name, temps in boiling_points.items():
    plt.plot(pressures_bar, temps, label=name)

# Set plot labels and title
plt.xlabel("Pressure (bar)")
plt.ylabel("Boiling Point (°C)")
plt.title("Boiling Point vs Pressure for Methanol, Ethanol, and Isopropanol")
plt.grid()
plt.legend()
plt.show()