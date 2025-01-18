from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
from matplotlib import pyplot as plt
import numpy as np
Pc = 200
eps = 10

MethanolWater = """
fuel CH3OH(L)   C 1 H 4 O 1
h,cal=-57040.0      t(k)=298.15       wt%=75.
oxid water H 2 O 1  wt%=25.
h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998
"""
EthanolWater = """
fuel C2H5OH(L)   C 2 H 6 O 1
h,cal=-66370.0      t(k)=298.15       wt%=75.
oxid water H 2 O 1  wt%=25.
h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998
"""
IPAWater = """
fuel C3H8O-2propanol C 3 H 8 O 1    wt%=75.
h,cal=-65133.     t(k)=298.15   rho=0.786
oxid water H 2 O 1  wt%=25.
h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998
"""
add_new_fuel( 'MethanolWater', MethanolWater )
add_new_fuel( 'EthanolWater', EthanolWater )
add_new_fuel( 'IPAWater', IPAWater )

fuellist = ["Ethanol", "Methanol", "Isopropanol", "RP1"]
#fuellist2 = ["EthanolWater", "MethanolWater", "IPAWater"]
colorlist = ['r', 'g', 'b', 'black']

for i, fuel in enumerate(fuellist):
    isplist = []
    isplist2 = []
    for MR in [0.2 + i*0.1 for i in range(50)]:
        C = CEA_Obj( oxName='LOX', fuelName=fuel)
        isplist.append(C.estimate_Ambient_Isp(Pc=Pc, MR=MR, eps=eps, Pamb=1)[0])
        #C = CEA_Obj( oxName='LOX', fuelName=fuellist2[i])
        #isplist2.append(C.estimate_Ambient_Isp(Pc=Pc, MR=MR, eps=eps, Pamb=1)[0])
    plt.plot([0.2 + i*0.1 for i in range(50)], isplist, label=fuel,color=colorlist[i], linestyle='solid')
    #plt.plot([0.5 + i*0.1 for i in range(50)], isplist2, label=f"{fuel} 25% Water",color=colorlist[i], linestyle='dashed')
    if i == 0:
        print(isplist2)
plt.legend()
plt.grid()
plt.xlabel("O/F")
plt.ylabel("ISP (s)")
plt.show()