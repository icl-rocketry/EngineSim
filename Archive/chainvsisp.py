import matplotlib.pyplot as plt
import numpy as np
from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
import matplotlib.colors as mcolors
# Define ChainLength values for the 10 lines
ChainLengths = np.linspace(1, 10, 10)
print(ChainLengths)
# Define MR values (50 points between 2 and 10)
MR_values = np.linspace(2, 10, 50)
PC = 25
EPS = 4.5

# Set up the color map
cmap = plt.get_cmap("viridis")  # You can choose other colormaps like 'plasma', 'cool', etc.
colors = cmap(np.linspace(0, 1, len(ChainLengths)))

# Create the plot
plt.figure(figsize=(10, 6))

for i, ChainLength in enumerate(ChainLengths):
    y_values = []
    fuelmix_str = (
    f'fuel CustomAlcohol   C {ChainLength} H {2 + 2 * ChainLength} O 1 \n'
    f'h,cal={ChainLength * -6602.136 -51247.814}      t(k)=298.15       wt%=100. \n'
    )
    add_new_fuel( 'fuelmix', fuelmix_str )
    for j, mr in enumerate(MR_values):
        C = CEA_Obj( oxName='N2O', fuelName="fuelmix")
        y_values.append(C.estimate_Ambient_Isp(Pc=PC, MR=mr, eps=EPS, Pamb=1)[0])
        #print(f"ChainLength = {ChainLength:.1f}, MR = {mr:.1f}, Isp = {y_values[j]:.1f}")
        #print(C.estimate_Ambient_Isp(Pc=PC, MR=mr, eps=EPS, Pamb=1)[0])
        
    plt.plot(MR_values, y_values, color=colors[i], label=f"ChainLength = {ChainLength:.1f}")

# Add labels, legend, and title
plt.xlabel("MR")
plt.ylabel("ISP (s)")
plt.title("Plot of ISP vs MR for different alcohol chain lengths")
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm = mcolors.Normalize(vmin=1, vmax=max(ChainLengths))), label="Number of carbons")
#plt.legend(loc="upper left", bbox_to_anchor=(0.5, 0.5))

# Show the plot
plt.show()