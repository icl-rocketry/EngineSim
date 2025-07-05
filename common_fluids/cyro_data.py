from pyfluids import Fluid, Input, FluidsList
import matplotlib.pyplot as plt
import numpy as np
from os import system

system('cls')

fluids_list = [
    {'name': 'Oxygen', 'fluid_type': FluidsList.Oxygen},
    {'name': 'Nitrogen', 'fluid_type': FluidsList.Nitrogen},
    {'name': 'Propane', 'fluid_type': FluidsList.nPropane}, 
    {'name': 'Methane', 'fluid_type': FluidsList.Methane},  
    {'name': 'Hydrogen', 'fluid_type': FluidsList.Hydrogen},
]

# Initialize fluids
fluids = {}
for fluid in fluids_list:
    fluids[fluid['name']] = {
        'fluid': Fluid(fluid['fluid_type']),
        'density': None,
        'specific_heat': None,
        'conductivity': None,
        }

P = np.concatenate([np.array([1, 5]), np.linspace(10, 100, 10)]) * 1e5  # Pressure range

# Calculate density data for each fluid
for name, fluid_info in fluids.items():
    fluid = fluid_info['fluid']
    
    T_fluid = np.linspace(fluid.min_temperature+1.1, 500-273.15, 500)
    
    # Initialize arrays for all properties
    fluid_info['density'] = np.zeros((len(P), len(T_fluid)))
    fluid_info['specific_heat'] = np.zeros((len(P), len(T_fluid)))
    fluid_info['conductivity'] = np.zeros((len(P), len(T_fluid)))
    fluid_info['temp_range'] = T_fluid
    
    # Calculate properties for all pressure-temperature combinations
    for j, pressure in enumerate(P):
        for i, temp in enumerate(T_fluid):
            fluid.update(Input.temperature(temp), Input.pressure(pressure))
            fluid_info['density'][j, i] = fluid.density
            fluid_info['specific_heat'][j, i] = fluid.specific_heat
            fluid_info['conductivity'][j, i] = fluid.conductivity

storage_pressure = 1e5 # atmospheric pressure
# storage_pressure = 4e5 # 3 bar (g)

# Print critical point and storage properties
print("Critical Point Properties:")
print("-" * 50)
for name, fluid_info in fluids.items():
    fluid = fluid_info['fluid']
    fluid.update(Input.pressure(storage_pressure), Input.quality(0))
    print(f'{name:<8} - Critical: T: {(fluid.critical_temperature+273.15):.2f} K / {fluid.critical_temperature:.2f} °C, P: {(fluid.critical_pressure/1e5):.2f} Bar')

print(f"\nStorage Properties at {storage_pressure/1e5:.1f} Bar (a):")
print("-" * 50)
for name, fluid_info in fluids.items():
    fluid = fluid_info['fluid']
    print(f'{name:<8}: {(fluid.temperature+273.15):.2f} K / {fluid.temperature:.2f} °C, {fluid.density:.2f} kg/m³') 

def create_property_plots(fluids, P, property_name, ylabel, title_suffix):
    """Create plots for a specific fluid property"""
    # Calculate subplot dimensions based on number of fluids
    num_fluids = len(fluids)
    cols = int(np.ceil(np.sqrt(num_fluids)))
    rows = int(np.ceil(num_fluids / cols))
    
    # Create subplot layout
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    
    # Handle case where there's only one subplot
    if num_fluids == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes
    else:
        axes = axes.flatten()
    
    # Create plots for each fluid
    for idx, (name, fluid_info) in enumerate(fluids.items()):
        ax = axes[idx]
        
        # Prepare data for plotting
        temp_range = fluid_info['temp_range'] + 273.15
        property_data = fluid_info[property_name]
        colors = plt.cm.viridis(np.linspace(0, 1, len(P)))
        
        for i, pressure in enumerate(P):
            ax.plot(temp_range, property_data[i], 
                    label=f'{pressure/1e5:.1f} bar', color=colors[i])
        
        ax.set_xlabel('Temperature (K)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{name} {title_suffix}')
        ax.legend(loc='best', fontsize='small')
        ax.set_xlim(min(temp_range), max(temp_range))
        ax.grid(True, alpha=0.3)
    
    # Hide any unused subplots
    for idx in range(num_fluids, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig

# Create plots for density, specific heat, and conductivity
density_fig = create_property_plots(fluids, P, 'density', 'Density (kg/m³)', 'Density vs Temperature')
specific_heat_fig = create_property_plots(fluids, P, 'specific_heat', 'Specific Heat (J/kg·K)', 'Specific Heat vs Temperature')
conductivity_fig = create_property_plots(fluids, P, 'conductivity', 'Thermal Conductivity (W/m·K)', 'Thermal Conductivity vs Temperature')

plt.show()