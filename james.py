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
    print(debugObj.get_full_cea_output(Pc=self.Pc, MR=self.MR, short_output=1, pc_units="bar",eps=[1,2,3]))

  def Array_print(self):
    ArrayObj = rocketcea.cea_obj.CEA_Obj(oxName = self.oxName, fuelName = self.fuelName)
    DataString = ArrayObj.get_full_cea_output(Pc=self.Pc, MR=self.MR, short_output=1, pc_units="bar",subar=[3,2],eps=[2,3])
    DataArray = DataString.split()
    #print(DataArray)
    n = 5 # The number here is the number of data u have
    P_Array = [0] * n
    T_Array = [0] * n
    RHO_Array = [''] * n
    M_Array = [0] * n
    Cp_Array = [0] * n
    Gamma_Array = [0] * n
    SON_VEL_Array = [0] * n
    i = 0
    self.success = False
    while not self.success:
      if DataArray[i] == 'P,':
        for j in range(n):
          P_Array[j] = float(DataArray[i + j + 2])
        self.success = True
      i = i + 1
    print(P_Array)
    self.success = False
    i = 0
    while not self.success:
      if DataArray[i] == 'T,':
        for j in range(n):
          T_Array[j] = float(DataArray[i + j + 2])
        self.success = True
      i = i + 1
    print(T_Array)
    self.success = False
    i = 0
    while not self.success:
      if DataArray[i] == 'RHO,':
        for j in range(n):
          RHO_Array[j] = DataArray[i + j + 2]
        self.success = True
      i = i + 1
    for i in range (n):
      self.Temp = RHO_Array[i].split('-')
      RHO_Array[i] = float(self.Temp[0]) * (10 ** (-int(self.Temp[1])))
    print(RHO_Array)
    self.success = False
    i = 0
    while not self.success:
      if DataArray[i] == 'M,':
        for j in range(n):
          M_Array[j] = float(DataArray[i + j + 2])
        self.success = True
      i = i + 1
    print(M_Array)
    self.success = False
    i = 0
    while not self.success:
      if DataArray[i] == 'Cp,':
        for j in range(n):
          Cp_Array[j] = float(DataArray[i + j + 2])
        self.success = True
      i = i + 1
    print(Cp_Array)
    self.success = False
    i = 0
    while not self.success:
      if DataArray[i] == 'GAMMAs':
        for j in range(n):
          Gamma_Array[j] = float(DataArray[i + j + 1])
        self.success = True
      i = i + 1
    print(Gamma_Array)
    self.success = False
    i = 0
    while not self.success:
      if DataArray[i] == 'SON':
        for j in range(n):
          SON_VEL_Array[j] = float(DataArray[i + j + 2])
        self.success = True
      i = i + 1
    print(SON_VEL_Array)



thanos = RocketEngine(
    oxName = "N2O",
    fuelName = "Methanol",
    thrust = 3000,
    Pc = 20,
    Pe = 0.85,
    MR = 3.5)

thanos.debug_print()
#thanos.Array_print()
