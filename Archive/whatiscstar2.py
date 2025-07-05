from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
import numpy as np
import rocketcea.cea_obj
from matplotlib import pyplot as plt
        
MR_low = 0.1
MR_high = 6
n_1 = 50

def newround(x):
    return round(x, -int(np.floor(np.log10(abs(x)))) + 4)
vectorizedround = np.vectorize(newround)


MRlist = np.linspace(MR_low, MR_high, n_1)
#waterlist = np.linspace(0, 25, 2)
waterlist = [0]
cstarlist = np.zeros(n_1)
for j, waterperc in enumerate(waterlist):
    fuelmix_str = (
    f'fuel C3H8O-2propanol C 3 H 8 O 1 \n'
    f'h,cal=-65133.     t(k)=298.15 wt%={100 - waterperc} \n'
    f'oxid water H 2 O 1  wt%={waterperc} \n'
    f'h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998 \n'
    )
    add_new_fuel( 'fuelmix', fuelmix_str )
    for i, MR in enumerate(MRlist):
        cstarlist[i] = CEA_Obj(oxName = "N2O", fuelName = "fuelmix",cstar_units = 'm/s',fac_CR=8.7273552404
                               ).get_Cstar(Pc=20, MR=MR)
    #poly = np.polyfit(MRlist, cstarlist, 10)
    plt.plot(MRlist, cstarlist, color=[0.2 + waterperc / 40, 0.2, 0.5] ,label=f'W%:{waterperc}')
    ofstring = ','.join(map(str,vectorizedround(MRlist)))
    cstarstring = ','.join(map(str,vectorizedround(cstarlist)))
    print(f'std::vector<float> cTableOF = {"{"}{ofstring}{"}"} \n\nstd::vector<float> cTableCStar = {"{"}{cstarstring}{"}"}')
plt.xlabel("O/F")
plt.ylabel("c* (m/s)")
plt.style.use('seaborn-v0_8-dark')
plt.grid()
plt.legend()
plt.show()