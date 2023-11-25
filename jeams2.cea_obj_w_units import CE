from rocketcea.cea_obj_w_units import CEA_Obj

from rocketcea.cea_obj_w_units import CEA_Obj
import numpy as np
import rocketcea.cea_obj
class RocketEngine:
  def __init__(self, oxName, fuelName, thrust, Pc, Pe, MR):
    self.oxName = oxName
    self.fuelName = fuelName
    self.Pc = Pc
    self.Pe = Pe
    self.MR = MR

    self.cea = CEA_Obj(oxName = oxName,
                       fuelName = fuelName,
                       isp_units='sec',
                       cstar_units = 'm/s',
                       pressure_units='Bar',
                       temperature_units='K',
                       sonic_velocity_units='m/s',
                       enthalpy_units='J/g',
                       density_units='kg/m^3',
                       specific_heat_units='J/kg-K',
                       viscosity_units='lbf-sec/sqin',  # wtf
                       thermal_cond_units='mcal/cm-K-s',  # wtf
                       fac_CR=None,
                       make_debug_prints=False)

  def debug_print(self):
    debugObj = rocketcea.cea_obj.CEA_Obj(oxName=self.oxName, fuelName=self.fuelName)
    print(debugObj.get_full_cea_output(Pc=self.Pc, MR=self.MR, short_output=1, pc_units="bar",subar=[1], fac_CR=5,show_transport=1))

  def Array_print(self):
    ArrayObj = rocketcea.cea_obj.CEA_Obj(oxName = self.oxName, fuelName = self.fuelName)
    DataString = ArrayObj.get_full_cea_output(Pc=self.Pc, MR=self.MR, short_output=1, pc_units="bar",subar=[1], fac_CR=5, show_transport=1)
    DataArray = DataString.split()
    #print(DataString)
    ##print(DataArray)



thanos = RocketEngine(
    oxName = "N2O",
    fuelName = "Methanol",
    thrust = 3000,
    Pc = 20,
    Pe = 0.85,
    MR = 3.5)

thanos.debug_print()
thanos.Array_print()
