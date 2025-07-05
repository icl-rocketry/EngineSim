from pyfluids import Fluid, FluidsList, Input, Mixture
from matplotlib import pyplot as plt
import numpy as np
from CoolProp.CoolProp import PropsSI

#print(PropsSI('C','P',20e5,'Q',1,'NitrousOxide') / PropsSI('O','P',20e5,'Q',1,'NitrousOxide'))
print(PropsSI('C','P',20e5,'Q',1,'NitrousOxide') / PropsSI('O','P',20e5,'Q',1,'NitrousOxide'))