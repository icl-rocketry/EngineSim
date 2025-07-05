# Plz Delet all the titles for the text file before using this program
import numpy as np
import matplotlib.pyplot as plt
import re

with open('Thanos_RPA.txt', 'r') as input_file:
    lines = input_file.readlines()

lines = lines[8:]
filtered_lines = [line for line in lines if not line.startswith('#')]

# Define a regular expression to match numbers
number_pattern = re.compile(r'[-+]?\d*\.\d+|\d+')

filtered_lines = []
for line in lines:
    # Extract numerical values from the line using the regular expression
    numbers = number_pattern.findall(line)
    # Join the numbers into a comma-separated string and add it to the filtered lines
    filtered_line = (' '.join(numbers) + '\n')
    if filtered_line.strip():
        filtered_lines.append(filtered_line)

with open('Thanos_Filtered.txt', 'w') as output_file:
    output_file.writelines(filtered_lines)

x = np.array([25, 50, 100, 150, 200, 250, 300, 350, 400])
y = np.array([77.6, 75.5, 72.8, 63.2, 60, 55, 45, 37, 28]) * 1e9
x2 = np.array([298, 323, 373, 423, 473, 523, 573, 623, 673]) - 273.15
y2 = np.array([204, 198, 181, 182, 158, 132, 70, 30, 12]) * 1e6

coefficients = np.polyfit(x, y, 2)
coefficients2 = np.polyfit(x2, y2, 2)

A = float(coefficients[0])
B = float(coefficients[1])
C = float(coefficients[2])
A2 = float(coefficients2[0])
B2 = float(coefficients2[1])
C2 = float(coefficients2[2])



a = 27 * 10 ** -6
k = 100
v = 0.33
t_w = 0.00075

file_path = 'Thanos_Filtered.txt' # name of the text file


pos = []
rad = []
twc = []
twg = []
tc = []
yieldstress = []
tempstress_t = []
tempstress_l = []
tempstress_p = []
von_mises = []
a_channel = []

qtotal = []

data_arrays = []

with open(file_path, 'r') as file:
    for line in file:
        line_array = line.split()
        line_array = [float(item) for item in line_array]
        data_arrays.append(line_array)

        T = float(line_array[6] - 273.15)
        if 0 < T < 400:
            E = A * (T ** 2) + B * T + C
            Ys = A2 * (T ** 2) + B2 * T + C2
        else:
            E = 0
            Ys = 0
        q = float(line_array[5]) * 1000
        stress_t = (E * a * q * t_w) / (2 * (1 - v) * k)
        stress_t2 = E * a * (float(line_array[6]) - float(line_array[8]))
        stress_p = 35e5 * 0.5 * (0.00475 * float(line_array[1]) / 47.13 / t_w) ** 2
        a_channel.append(0.00475 * float(line_array[1]) / 47.13)
        
        qtotal.append(float(line_array[5]))
        twg.append(float(line_array[6]))
        twc.append(float(line_array[8]))
        tc.append(float(line_array[9]))
        #print(float(line_array[6]) - float(line_array[8]))
        line_array.append(stress_t)
        
        pos.append(float(line_array[0]) / 1000)
        rad.append(float(line_array[1]) / 1000)
        
        yieldstress.append(Ys * 1e-6)
        tempstress_t.append(stress_t * 1e-6)
        tempstress_l.append(stress_t2 * 1e-6)
        tempstress_p.append(stress_p* 1e-6)
        s1 = stress_t + stress_p
        s2 = stress_t2
        von_mises.append(np.sqrt(0.5 * ((s1 - s2)**2 + (s2)**2 + (s1)**2)) * 1e-6)



#for array in data_arrays:
#    print(array)
#print(line_array)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
#ax.plot(pos, rad)
ax[0].set_ylabel("Chamber Radius (m)")
ax2 = ax[0]
ax2.plot(pos, yieldstress, color="tab:green"    , label="Yield stress")
ax2.plot(pos, tempstress_t, color="tab:pink"    , label="Tangential Thermal")
ax2.plot(pos, tempstress_l, color="tab:purple"  , label="Longitudinal Thermal")
ax2.plot(pos, tempstress_p, color="tab:orange"  , label="Tangential Pressure")
ax2.plot(pos, von_mises, color="tab:red"        , label="Von-Mises")
ax2.set_ylabel("Stress (MPA)")
ax2.set_xlabel("20Bar 10%Film 3.3COF Bartz")
ax[0].grid()
ax2.legend()
ax3 = ax[1]
ax3.plot(pos, twg, label="twg")
ax3.plot(pos, twc, label='twc')
ax3.plot(pos, tc, label='tc')
ax3.legend()
ax3.grid()

#total_power = 0
#for i, q_i in enumerate(qtotal):
#    total_power += q_i * 2 * rad[i] * np.pi * (pos[1] - pos[0]) * 1000
#print(total_power)
#print((tc[0] - tc[-1]) * 2530)