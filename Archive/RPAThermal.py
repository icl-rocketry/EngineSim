from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
print(f"Ethanol Qs {PropsSI('H','P',101325,'Q',1,'Ethanol')-PropsSI('H','P',101325,'Q',0,'Ethanol')}")


#print(f"Ethanol Ts {PropsSI('T','P',101325,'Q',0,'Ethanol')}")
P = [1e5, 10e5, 40e5]
T = 270
fluid = 'Ethanol'
for _, p in enumerate(P):
    Cp = PropsSI('C','P',p,'T',T,fluid)
    D = PropsSI('D','P',p,'T',T,fluid)
    V = PropsSI('V','P',p,'T',T,fluid)
    L = PropsSI('L','P',p,'T',T,fluid)
    Ts = PropsSI('T','P',p,'Q',0,fluid)
    Qs = (PropsSI('H','P',p,'Q',1,fluid)-PropsSI('H','P',p,'Q',0,fluid))/1000
    print(f"{p/1e6} {T} {Cp:.2f} {D:.2f} {V:.2f} {L:.2f} {Ts:.2f} {Qs:.2f}")
    
Pvap = []
Tvap = []
for p in range(100000, 4000000, 100000):
    Pvap.append(p)
    Tvap.append(PropsSI('T','P',p,'Q',0,fluid))
plt.plot(Pvap, Tvap)