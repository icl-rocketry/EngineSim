from pyfluids import Fluid, FluidsList, Input, Mixture
from matplotlib import pyplot as plt
import numpy as np

pressurelist = np.linspace(1e5, 80e5, 100)
c = np.zeros(100)
for i, pressure in enumerate(pressurelist):
  c[i] = PropsSI('C','P',pressure,'T',293,'NitrousOxide') / PropsSI('O','P',pressure,'T',293,'NitrousOxide')
plt.plot(templist, c)
plt.show()
#test = Fluid(FluidsList.Methanol).with_state(Input.pressure(3e1), Input.temperature(25));
#test.sp

templist = np.linspace(273, 373, 100)
c = np.zeros(100)
for i, temp in enumerate(templist):
  c[i] = PropsSI('C','P',40e5,'T',temp,'NitrousOxide') / PropsSI('O','P',40e5,'T',temp,'NitrousOxide')
plt.plot(templist, c)
plt.show()

PropsSI('C','P',pressure,'T',293,'NitrousOxide') / PropsSI('O','P',pressure,'T',293,'NitrousOxide')

from pyfluids import Fluid, FluidsList, Input, Mixture
from matplotlib import pyplot as plt
import numpy as np

PropsSI('C','P',20e5,'Q',1,'NitrousOxide') / PropsSI('O','P',20e5,'Q',1,'NitrousOxide')