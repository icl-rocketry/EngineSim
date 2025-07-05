import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

#AlSi10Mg
x1 = np.array([298, 323, 373, 423, 473, 523, 573, 623, 673]) - 273.15
y1 = np.array([204, 198, 181, 182, 158, 132, 70, 30, 12])
#Inconel 718
x2 = np.array([93, 204, 316, 427, 538, 649, 760]) # Ys
y2 = np.array([1172, 1124, 1096, 1076, 1069, 1027, 758]) # Ys
#ABD900 https://uploads-ssl.webflow.com/5e57c6d0de09e9f96a2acb75/611284157f1f743e4c1d257a_Alloyed_ABD%C2%AE-900AM%20datasheet.pdf
x3 = np.array([29,225,440,599,755,843,873,917])
y3 = np.array([1090,1028,976,937,897,883,836,711])
#GRCop-42
x4 = np.array([300, 400, 500, 600, 700, 800, 900, 1000]) - 273.15 # Ys Temps
y4 = np.array([175, 170, 160, 150, 135, 120, 95, 70]) # Ys
# FS-85 https://tanb.org/view/tailoring-nb-based-alloys-for-additive-manufacturing--from-powder-production-to-parameter-optimization
x5 = np.array([22.22, 600, 895.24, 1100.53, 1292.06, 1451.85, 1590.48])  
y5 = np.array([658.30, 491.52, 449.83, 296.68, 210.88, 141.92, 72.97])  


# Step 1: Define linear interpolation functions for each dataset
interp1 = interp1d(x1, y1, kind='linear')
interp2 = interp1d(x2, y2, kind='linear')
interp3 = interp1d(x3, y3, kind='linear')
interp4 = interp1d(x4, y4, kind='linear')
interp5 = interp1d(x5, y5, kind='linear')

# Step 2: Create a fine x-axis to plot smooth interpolated lines
x_fine1 = np.linspace(x1.min(), x1.max(), 500)
x_fine2 = np.linspace(x2.min(), x2.max(), 500)
x_fine3 = np.linspace(x3.min(), x3.max(), 500)
x_fine4 = np.linspace(x4.min(), x4.max(), 500)
x_fine5 = np.linspace(x5.min(), x5.max(), 500)

# Step 3: Plot the original data points and the interpolated lines
plt.figure(figsize=(10, 6))

plt.plot(x1, y1, 'o', label="AlSi10Mg", color="blue")
plt.plot(x_fine1, interp1(x_fine1), '--', color="blue")

plt.plot(x2, y2, 'o', label="Inconel 718", color="green")
plt.plot(x_fine2, interp2(x_fine2), '--', color="green")

plt.plot(x3, y3, 'o', label="ABD900-AM", color="gray")
plt.plot(x_fine3, interp3(x_fine3), '--', color="gray")

plt.plot(x4, y4, 'o', label="GRCop-42", color="red")
plt.plot(x_fine4, interp4(x_fine4), '--', color="red")

plt.plot(x5, y5, 'o', label="FS-85", color="purple")
plt.plot(x_fine5, interp5(x_fine5), '--', color="purple")


plt.xlabel('Temperature (C)')
plt.ylabel('Yield Stress (MPa)')
plt.title('Yield stress vs Temperature of various AM metals')
plt.legend()
plt.grid()
plt.show()