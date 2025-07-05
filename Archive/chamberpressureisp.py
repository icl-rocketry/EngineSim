from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
from matplotlib import pyplot as plt
import numpy as np

eps = 6
fuellist = ["Ethanol", "Methanol", "Isopropanol", "RP1"]
colorlist = ['r', 'g', 'b', 'black']
plt.figure()
for Pc in range(20, 41, 5):
    for i, fuel in enumerate(fuellist):
        isplist = []
        for MR in [0.2 + i*0.1 for i in range(50)]:
            C = CEA_Obj(oxName='N2O', fuelName=fuel)
            isplist.append(C.estimate_Ambient_Isp(Pc=Pc, MR=MR, eps=eps, Pamb=1)[0])
        plt.plot([0.2 + i*0.1 for i in range(50)], isplist, label=fuel + " " + str(Pc), color=colorlist[i], linestyle='solid')
    plt.legend()
    plt.grid()
    plt.xlabel("O/F")
    plt.ylabel("ISP (s)")
    plt.title(f"Chamber Pressure: {Pc} psi")
plt.show()
