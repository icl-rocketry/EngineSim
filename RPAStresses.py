# Plz Delet all the titles for the text file before using this program
import numpy as np
import matplotlib.pyplot as plt

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

def calculate_stress_p(P_l, P_g, w, t_w, E, a, q, v, k):
    stress_t = 0.5 * (P_l - P_g) * ((w - t_w) ** 2)
    return stress_t

def calculate_stress_l(E, a, delta_T):
    stress_l = E * a * delta_T
    return stress_l

def calculate_stress_t_fornow(t_w, E, a, q, v, k):
    stress_t = (E * a * q * t_w) / (2 * (1 - v) * k)
    return stress_t

a = 27 * 10 ** -6
k = 100
v = 0.33
t_w = 0.0015

file_path = 'Thanos_RPA_calcs.txt' # name of the text file


pos = []
rad = []
yieldstress = []
tempstress_t = []
tempstress_l = []
tempstress_p = []
von_mises = []

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
        stress_t = calculate_stress_t_fornow(t_w, E, a, q, v, k)
        stress_t2 = E * a * (float(line_array[6]) - float(line_array[8]))
        stress_p = 35e5 * 0.5 * (0.00475 * float(line_array[1]) / 51.1600 / t_w) ** 2
        print(float(line_array[6]) - float(line_array[8]))
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
fig, ax2 = plt.subplots(1, 1, sharey=True)
#ax.plot(pos, rad)
#ax.set_ylabel("Chamber Radius (m)")
#ax2 = ax.twinx()
ax2.plot(pos, yieldstress, color="tab:green"    , label="Yield stress")
ax2.plot(pos, tempstress_t, color="tab:pink"    , label="Tangential Thermal")
ax2.plot(pos, tempstress_l, color="tab:purple"  , label="Longitudinal Thermal")
ax2.plot(pos, tempstress_p, color="tab:orange"  , label="Tangential Pressure")
ax2.plot(pos, von_mises, color="tab:red"        , label="Von-Mises")
ax2.set_ylabel("Stress (MPA)")
ax2.legend()