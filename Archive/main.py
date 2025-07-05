from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
import numpy as np
import rocketcea.cea_obj
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar
from pint import UnitRegistry
from pyfluids import Fluid, FluidsList, Input
import pandas as pd

def lininterp(x, old_min, old_max, new_min, new_max):
  return (x - old_min)/(old_max - old_min) * (new_max - new_min) + new_min

def pointinterp(x, start, end):
  return np.array([start[0] + x * (end[0] - start[0]), start[1] + x * (end[1] - start[1])])

class RocketEngine:
  def __init__(self, oxName, fuelName, thrust, Pc, Pe, MR, radius_cylinder):
    self.oxName = oxName
    self.fuelName = fuelName
    self.Pc = Pc
    self.Pe = Pe
    self.MR = MR
    self.radius_cylinder = radius_cylinder

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
                       viscosity_units='lbf-sec/sqin',      # wtf
                       thermal_cond_units='mcal/cm-K-s',  # wtf
                       fac_CR=None,
                       make_debug_prints=False)

    self.eps = self.cea.get_eps_at_PcOvPe(Pc=Pc, MR=MR, PcOvPe = Pc/Pe)
    self.Tc, self.Tt, self.Te = self.cea.get_Temperatures(Pc=Pc, MR=MR, eps=self.eps)
    self.Pt = self.Pc / self.cea.get_Throat_PcOvPe(Pc=Pc, MR=MR)
    self.ve = self.cea.get_SonicVelocities(Pc=Pc,MR=MR,eps=self.eps)[2] * self.cea.get_MachNumber(Pc=Pc, MR=MR, eps=self.eps)
    self.mdot = thrust / self.ve
    self.mwt, self.gt = self.cea.get_Throat_MolWt_gamma(Pc=Pc, MR=MR, eps=self.eps)
    self.Rt = 8314.46 / self.mwt
    self.At = self.mdot / (self.Pt * 1e5) * np.sqrt(self.Rt * self.Tt / self.gt)
    self.CR = radius_cylinder**2 * np.pi / self.At
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
                    viscosity_units='lbf-sec/sqin',      # wtf
                    thermal_cond_units='mcal/cm-K-s',  # wtf
                    fac_CR=self.CR,
                    make_debug_prints=False)

    self.eps = self.cea.get_eps_at_PcOvPe(Pc=Pc, MR=MR, PcOvPe = Pc/Pe)
    self.Tc, self.Tt, self.Te = self.cea.get_Temperatures(Pc=Pc, MR=MR, eps=self.eps)
    self.Pt = self.Pc / self.cea.get_Throat_PcOvPe(Pc=Pc, MR=MR)
    self.ve = self.cea.get_SonicVelocities(Pc=Pc,MR=MR,eps=self.eps)[2] * self.cea.get_MachNumber(Pc=Pc, MR=MR, eps=self.eps)
    self.mdot = thrust / self.ve

    self.mwt, self.gt = self.cea.get_Throat_MolWt_gamma(Pc=Pc, MR=MR, eps=self.eps)
    self.mwe, self.ge = self.cea.get_exit_MolWt_gamma(Pc=Pc, MR=MR, eps=self.eps)
    self.Rt = 8314.46 / self.mwt
    self.Re = 8314.46 / self.mwe

    self.Cp_c, self.Cp_t, self.Cp_e = self.cea.get_HeatCapacities(Pc=Pc, MR=MR, eps=self.eps)
    self.density_c = self.cea.get_Chamber_Density(Pc=Pc, MR=MR, eps=self.eps)

    _, self.visc, self.k, self.pr = self.cea.get_Chamber_Transport(Pc=Pc, MR=MR, eps=self.eps)
    self.cstar = self.cea.get_Cstar(Pc=Pc, MR=MR)
    self.isp = self.cea.estimate_Ambient_Isp(Pc=Pc, MR=MR, eps=self.eps, Pamb=1)

    self.At = self.mdot / (self.Pt * 1e5) * np.sqrt(self.Rt * self.Tt / self.gt)
    self.Ae = self.At * self.eps
    self.Ac = radius_cylinder**2 * np.pi
    self.CR = self.Ac / self.At

  def debug_print(self):
      debugObj = rocketcea.cea_obj.CEA_Obj(oxName=self.oxName, fuelName=self.fuelName, fac_CR=self.CR)
      print(debugObj.get_full_cea_output(Pc=self.Pc, MR=self.MR, eps=self.eps, short_output=1, pc_units="bar"))

class DimensionedRocketEngine:
  def __init__(self, oxName, fuelName, Pc, Pe, MR, Ac, At, Ae):
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
                  viscosity_units='lbf-sec/sqin',      # wtf
                  thermal_cond_units='mcal/cm-K-s',  # wtf
                  fac_CR=self.Ac/self.At,
                  make_debug_prints=False)
    self.oxName = oxName
    self.fuelName = fuelName
    self.Pc = Pc
    self.Pe = Pe
    self.MR = MR
    self.radius_cylinder = np.sqrt(self.Ac / np.pi)

class Metal:
  def __init__(self, k, a, yield_array, modulus_array):
    self.k = k
    self.a = a
    self.yield_array = yield_array
    self.modulus_array = modulus_array
  def yield_stress(self, T):
    return np.interp(T, self.yield_array[:,0], self.yield_array[:,1])
  def modulus(self, T):
    return np.interp(T, self.modulus_array[:,0], self.modulus_array[:,1])
  
class Contour:
  def __init__(self, engine, chamber_length, radius_cylinder, points):
    rt = np.sqrt(engine.At / np.pi)
    re = np.sqrt(engine.Ae / np.pi)
    Le = 0.8 * rt * (np.sqrt(engine.eps) - 1) / np.tan(15 * np.pi / 180)
    total_length = chamber_length + Le

    Tn = 22 * np.pi / 180
    Te = 14 * np.pi / 180

    # Cyl (For my laziness assume R2 = 0)
    b = 45 * np.pi / 180
    x_conv = chamber_length + 1.5 * rt * np.cos(-b - np.pi / 2)
    y_conv = 2.5 * rt + 1.5 * rt * np.sin(-b - np.pi / 2)

    x_slope = (radius_cylinder - y_conv) / np.tan(b)

    contour_cyl = np.array([
        [0, radius_cylinder],
        [x_conv - x_slope,radius_cylinder],
        [x_conv,y_conv]])

    # Converge -> Throat

    contour_converge_throat = np.zeros((points - 1,2))
    for i in range(1, points):
      t = lininterp(i, 0, points - 1, -b - np.pi / 2, - np.pi / 2)
      contour_converge_throat[i - 1] = np.array([chamber_length + 1.5 * rt * np.cos(t), 2.5 * rt + 1.5 * rt * np.sin(t)])

    # Throat -> Parabola

    contour_throat_parabola = np.zeros((points - 1,2))
    for i in range(1, points):
      t = lininterp(i, 0, points - 1, 0 - np.pi / 2, Tn - np.pi / 2)
      contour_throat_parabola[i - 1] = np.array([chamber_length + 0.382 * rt * np.cos(t), 1.382 * rt + 0.382 * rt * np.sin(t)])

    # Parabola
    parabola_p0 = contour_throat_parabola[-1]
    parabola_p2 = np.array([total_length, re])
    parabola_m1 = np.tan(Tn)
    parabola_m2 = np.tan(Te)
    parabola_c1 = parabola_p0[1] - parabola_m1 * parabola_p0[0]
    parabola_c2 = parabola_p2[1] - parabola_m2 * parabola_p2[0]
    parabola_p1 = np.array([(parabola_c2 - parabola_c1)/(parabola_m1 - parabola_m2),(parabola_m1 * parabola_c2 - parabola_m2 * parabola_c1)/(parabola_m1 - parabola_m2)])

    contour_parabola = np.zeros((points - 1,2))
    for i in range(1, points):
      t = lininterp(i, 0, points - 1, 0, 1)
      parabola_linear0 = pointinterp(t, parabola_p0, parabola_p1)
      parabola_linear1 = pointinterp(t, parabola_p1, parabola_p2)
      contour_parabola[i - 1] = pointinterp(t, parabola_linear0, parabola_linear1)

    contour = np.concatenate([contour_cyl, contour_converge_throat, contour_throat_parabola, contour_parabola])
    self.x = np.linspace(contour[:,0].min(), contour[:,0].max(), num=points)
    self.r = np.interp(self.x, contour[:,0], contour[:,1])

    self.throat = (chamber_length, rt)
    self.exit = (total_length, re)
    self.rc = radius_cylinder
    self.parabola_p1 = parabola_p1
    self.points = points

ureg = UnitRegistry()
Q_ = ureg.Quantity


class Thermals:
  def __init__(self, engine, contour, transport_case, hg_multiplier):
    self.engine = engine
    self.contour = contour

    x_choke = np.argmin(contour.r)

    #Finding mach number, https://kyleniemeyer.github.io/gas-dynamics-notes/compressible-flows/isentropic.html#equation-eq-area-ratio-loss
    def area_function(mach, area, gamma):

      area_ratio = area / engine.At
      if mach == 0:
        mach = 0.000001
      return (
          area_ratio - (
              (1.0/mach) * ((1 + 0.5*(gamma-1)*mach*mach) /
              ((gamma + 1)/2))**((gamma+1) / (2*(gamma-1)))
              )
          )

    points = contour.points
    self.M = np.zeros(points)
    self.T = np.zeros(points)
    self.density = np.zeros(points)
    self.hg1 = np.zeros(points)
    self.hg2 = np.zeros(points)
    self.hg3 = np.zeros(points)
    self.stag_recovery = np.zeros(points)
    #self.density = np.zeros(points)

    Cv = engine.Cp_c - engine.Rt
    rt = np.sqrt(engine.At / np.pi)

    for i in range(0,contour.points):


      #Transport case:
      # 0 - throat transports until throat, then linearly interpolate to exit at exit
      # 1 - throat transports
      # 2 - exit transports
      if transport_case == 0:
        if i < x_choke:
          gamma = engine.gt
        else:
          gamma = lininterp(i, x_choke, contour.points, engine.gt, engine.ge)
      elif transport_case == 1:
        gamma = engine.gt
      elif transport_case == 2:
        gamma = engine.ge

      area = np.pi * contour.r[i] * contour.r[i]
      if i == x_choke:
        M = 1
      elif i < x_choke:
        M = root_scalar(area_function, args=(area, gamma), bracket=[0, 1]).root
      else:
        M = root_scalar(area_function, args=(area, gamma), bracket=[1, 10]).root
      self.M[i] = M

      #T = engine.Tc - engine.mdot * engine.mdot / (2 * Cv * engine.density_c * engine.density_c * area * area) * (1 + 0.5 * (engine.gt - 1) * M * M)**(2 / (engine.gt - 1))
      T = engine.Tc / (1 + 0.5 * (gamma - 1) * M*M)
      self.T[i] = T
      density = engine.density_c * (1 + 0.5 * (gamma - 1) * M*M)**(1/(gamma - 1))
      self.density[i] = density
      correction_factor = 1 / ((0.5 * T / engine.Tc * (1 + 0.5 * (gamma - 1) * M**2)+0.5)**0.68 * (1 + 0.5 * (gamma - 1) * M**2)**0.12)


      r = contour.r[i]
      Cp = engine.Cp_c
      visc = engine.visc
      pr = engine.pr

      self.stag_recovery[i] = (1 + 0.5*(gamma - 1)*M**2*pr**0.33)

      #Bartz

      bPc = Q_(engine.Pc, ureg.bar).to('psi')
      bcstar = Q_(engine.cstar, ureg.meter / ureg.second).to('feet / second')
      bDt = Q_(np.sqrt(4 * engine.At / np.pi), ureg.meter).to('inch')
      bcurv = Q_((1.5 * rt + 0.382 * rt) * 0.5, ureg.meter).to("inch")
      bCp = Q_(Cp, ureg.joule / (ureg.kilogram * ureg.degK)).to('Btu / (lb * delta_degF)')
      bvisc = Q_(visc, ureg.lbf * ureg.second/ (ureg.inch ** 2))
      bg = Q_(9.81, ureg.meter / ureg.second**2).to('feet / second**2')
      bAtA = engine.At / area
      self.hg1[i] = Q_((0.026 / bDt**0.2 * (bvisc**0.2 * bCp / pr**0.6) * (bPc * bg / bcstar)**0.8 * (bDt / bcurv)**0.1 * bAtA**0.9 * correction_factor).magnitude, "Btu / (inch ** 2 * second)").to("W / (m**2)").magnitude * hg_multiplier

      #Adami
      R = engine.Rt
      Z = np.pi * contour.r[0]**2 / (2 * np.pi * contour.r[0] * contour.x[-1])
      self.hg2[i] = Z * engine.mdot / (2 * area) * Cp * visc**0.3 * pr **(2/3)
      
      self.hg3[i] = 0.023 * (density * M * np.sqrt(gamma * R * T))**0.8 / (2 * r)**0.2 * pr**0.4 * engine.k / visc**0.8

class Channels:
  def __init__(self, engine, contour, thermals, h, hc0, hcmultiplier, a0, N): #h = wall thickness
    points = contour.points
    self.hc = np.zeros(points)
    self.a = np.zeros(points)
    self.A = np.zeros(points)
    self.per = np.zeros(points)

    r_list = contour.r
    r0 = r_list[0]
    self.N = N
    self.h = h

    for i, r in enumerate(r_list):
      hc = hc0 * hcmultiplier[i]
      a = a0 * (r + h) / (r0 + h)
      Area = hc0 * a

      a2 = a0 * (r + h + hc) / (r0 + h)

      self.hc[i] = hc0
      self.a[i] = a
      self.A[i] = hc * a
      self.per[i] = 2 * hc0 + a + a2

class ChannelSim:
  def __init__(self, engine, contour, thermals, channel, metal):
    self.metal = metal
    mdot = engine.mdot / (channel.N * (engine.MR + 1))
    #mdot = 0.5 / channel.N


    points = contour.points
    self.Taw = np.zeros(points)
    self.Twg = np.zeros(points)
    self.Twc = np.zeros(points)
    self.Tco = np.zeros(points)
    self.velocity = np.zeros(points)
    self.hg = np.zeros(points)
    self.hc = np.zeros(points)
    self.hw = np.zeros(points)
    self.q = np.zeros(points)
    self.stress_pressure = np.zeros(points)
    self.stress_temperature = np.zeros(points)
    self.stress_temperature2 = np.zeros(points)
    self.stress_total = np.zeros(points)
    self.Pco = np.zeros(points)
    Tco_i = 300
    Pco_i = 40e5

    for i in range(contour.points-1, -1, -1):
      r = contour.r[i]
      T_inf = thermals.T[i]
      stag_recovery = thermals.stag_recovery[i]
      Taw = T_inf * 0.923
      hg = thermals.hg1[i]

      a = channel.a[i]
      A = channel.A[i]
      per = channel.per[i]

      methanol = Fluid(FluidsList.Methanol).with_state(Input.pressure(Pco_i), Input.temperature(Tco_i-275.15))

      k = methanol.conductivity
      density = methanol.density
      pr = methanol.prandtl
      viscosity = methanol.dynamic_viscosity
      cp = methanol.specific_heat

      Dh = 4 * A / per
      #velocity = mdot/(A * density * np.cos(30 * np.pi / 180))
      #velocity = mdot/(A * density * np.sin(np.arctan(204 / (75 * np.pi) * 0.075 / r)))
      velocity = mdot/(A * density)

      Twc = Tco_i + 100
      for j in range(0, 10):
        hc = 0.023 * k / Dh * (density * velocity * Dh / viscosity)**0.8 * pr ** 0.4 * (Twc / Tco_i)**-0.3
        H = 1 / (1 / hg + channel.h / metal.k + 1 / hc)
        #H = 1 / (1 / hg + 0.0001/1 + channel.h / metal.k + 1 / hc)
        q = H * (Taw - Tco_i)
        Twc = Tco_i + q / hc

      Twg = Twc + q * channel.h / metal.k

      stress_pressure = 0.5 * (40e5 - 1e5) * (a / channel.h) **2;
      stress_temperature = metal.modulus(Twg) * metal.a * q * channel.h * 0.5 / ((1 - 0.3) * metal.k)
      stress_temperature2 = metal.modulus(Twg) * metal.a * (Twg - Twc)

      #cf = 0.08 # 70e-6 / 0.0015 -> Moody chart
      rz = 60e-6
      cf = (-2*np.log10(0.27*rz/Dh))**-2


      #     Pr=; #Prandtl's Number
      # Cpg=; #Specific Heat of main gas in Combustion Chamber
      # Cpc=; #Specific Heat of Coolant
      # x=; #Downstream distance from injection point
      # s=; #Injection slot height
      # rho_c=; # Coolant Density
      # rho_g= # Combustion chamber gas density
      # v_c=; #Coolant Velocity
      # v_g=; #Combustion chamber gas velocity
      # Re_c=; #Reynold's Number of coolant
      # M=(rho_c*v_c)/(rho_g*v_g);

      # effectiveness=(0.83*(Pr)^(2/3))/(1.11+0.329*(Cpg/Cpc)*(((x/s)*(1/M))^1.43)*Re_c^-0.25); #Film cooling effectiveness
    
      # Taw=Twg-effectiveness*(Twg-Tco_i) ; #Taw is adiabatic wall temperature after cooling, Twg is adiabatic wall temperature without film cooling
      # # Tco_i is initial coolant temperature,

      self.Taw[i] = Taw
      self.Twg[i] = Twg
      self.Twc[i] = Twc
      self.Tco[i] = Tco_i
      self.velocity[i] = velocity
      self.hg[i] = hg
      self.hc[i] = hc
      self.hw[i] = metal.k / channel.h
      self.q[i] = q
      self.stress_pressure[i] = stress_pressure
      self.stress_temperature[i] = stress_temperature
      self.stress_temperature2[i] = stress_temperature2
      self.stress_total[i] = stress_pressure + stress_temperature + stress_temperature2
      self.Pco[i] = Pco_i

      dx = contour.x[1] - contour.x[0]
      Tco_i = Tco_i + 1 / (mdot * cp) * q * (2 * np.pi * r * dx / channel.N)
      Pco_i = Pco_i - 32 * cf * dx * mdot **2 / (density * np.pi**2 * Dh**5)

def displaysim(showtext):
  if showtext:
    print(f'Mass flux (kg/s) = {thanos.mdot}')
    print(f'Throat      (mm) = {thanos_contour.throat}')
    print(f'Exit        (mm) = {thanos_contour.exit}')
    print(f'Cylinder r  (mm) = {thanos_contour.rc}')
    print(f'Parabola    (mm) = {thanos_contour.parabola_p1}')
    print(f'ISP         (mm) = {thanos.isp}')
    print(f'CR         (mm) = {thanos.CR}')

  fig, ax = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
  #fig.set_size_inches(8, 5)

  color = 'tab:gray'
  ax[0].set_xlabel('position (m)')
  ax[0].set_ylabel('chamber radius (m)', color=color)
  ax[0].plot(thanos_contour.x, thanos_contour.r, color=color)
  ax[0].tick_params(axis='y', labelcolor=color)
  # ax[0].set_aspect('equal', adjustable='box')
  ax[0].set_ylim(0, 0.14)

  color = 'tab:blue'
  ax0_2 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
  ax0_2.set_ylabel('temperature (K)', color=color)  # we already handled the x-label with ax1
  ax0_2.plot(thanos_contour.x, thanos_thermals.T, color=color)
  ax0_2.spines['right'].set_position(('outward', 60))
  ax0_2.tick_params(axis='y', labelcolor=color)

  color = 'tab:orange'
  ax0_3 = ax[0].twinx()  # instantiate a second axes that shares the same x-axis
  ax0_3.set_ylabel('hg', color=color)  # we already handled the x-label with ax1
  line0_1, = ax0_3.plot(thanos_contour.x, thanos_thermals.hg1, color=color, label='Bartz')
  line0_2, = ax0_3.plot(thanos_contour.x, thanos_thermals.hg2, color=color, label='Adami')
  #line0_3, = ax0_3.plot(thanos_contour.x, thanos_thermals.hg3, color=color, label='Nusselt')
  ax0_3.tick_params(axis='y', labelcolor=color)
  ax0_3.set_yticks(np.arange(0, 3000, 500))
  fig.tight_layout()
  ax[0].grid()
  ax0_3.legend(handles=[line0_1, line0_2], loc='upper right')
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
  color = 'tab:gray'
  ax[1].set_xlabel('position (m)')
  ax[1].set_ylabel('chamber radius (m)', color=color)
  ax[1].plot(thanos_contour.x, thanos_contour.r, color=color)
  #ax[1].plot(thanos_contour.x, thanos_contour.r + thanos_channel.h, color=color)
  #ax[1].plot(thanos_contour.x, thanos_contour.r + thanos_channel.h + thanos_channel.hc, color=color)
  ax[1].tick_params(axis='y', labelcolor=color)
  #ax[1].set_ylim(0, 0.14)
  ax[1].set_box_aspect(1)

  color = 'tab:red'
  ax1_2 = ax[1].twinx()  # instantiate a second axes that shares the same x-axis
  ax1_2.set_ylabel('Hot Gas Wall Temperature (K)', color=color)  # we already handled the x-label with ax1
  #ax1_2.spines['right'].set_position(('outward', 60))
  ax1_2.tick_params(axis='y', labelcolor=color)
  line1_1, = ax1_2.plot(thanos_contour.x, channelSim.Twg, color='tab:red', label='T Combustion Side Wall')
  line1_2, = ax1_2.plot(thanos_contour.x, channelSim.Twc, color='tab:orange', label='T Coolant Side Wall')
  line1_3, = ax1_2.plot(thanos_contour.x, channelSim.Tco, color='tab:blue', label='T Coolant Bulk')
  ax[1].grid()
  ax1_2.legend(handles=[line1_1, line1_2, line1_3], loc='upper left')
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
  color = 'tab:gray'
  ax[2].set_xlabel('position (m)')
  ax[2].set_ylabel('chamber radius (m)', color=color)
  ax[2].plot(thanos_contour.x, thanos_contour.r, color=color)
  ax[2].tick_params(axis='y', labelcolor=color)
  ax[2].set_ylim(0, 0.14)
  ax2_2 = ax[2].twinx()  # instantiate a second axes that shares the same x-axis
  color = 'tab:orange'
  ax2_2.set_ylabel('Stress (MPa)', color=color)  # we already handled the x-label with ax1
  ax2_2.tick_params(axis='y', labelcolor=color)
  line2_1, = ax2_2.plot(thanos_contour.x, channelSim.stress_total * 1e-6, color='tab:pink', label='Total Stress')
  line2_3, = ax2_2.plot(thanos_contour.x, channelSim.stress_pressure * 1e-6, color='tab:purple', label='Pressure Stress')
  line2_2, = ax2_2.plot(thanos_contour.x, channelSim.metal.yield_stress(channelSim.Twg) * 1e-6, color='tab:green', label='Yield Stress')

  #ax1.set_aspect('equal', adjustable='box')
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  #ax3.set_yticks(np.arange(0, 3000, 500))
  ax[2].grid()
  ax2_2.legend(handles=[line2_1, line2_2, line2_3], loc='upper left')
  #----------------------------------------------------------------------------------------------------------------------------------------------------------------
  color = 'tab:gray'
  ax[3].set_xlabel('position (m)')
  ax[3].set_ylabel('chamber radius (m)', color=color)
  ax[3].plot(thanos_contour.x, thanos_contour.r, color=color)
  ax[3].tick_params(axis='y', labelcolor=color)
  # ax[2].set_aspect('equal', adjustable='box')
  ax[3].set_ylim(0, 0.14)

  color='tab:orange'
  ax3_2 = ax[3].twinx()  # instantiate a second axes that shares the same x-axis
  ax3_2.set_ylabel('Velocity (m/d)', color=color)  # we already handled the x-label with ax1
  ax3_2.tick_params(axis='y', labelcolor=color)
  line3_1, = ax3_2.plot(thanos_contour.x, channelSim.velocity, color=color)

  color='tab:pink'
  ax3_3 = ax[3].twinx()  # instantiate a second axes that shares the same x-axis
  ax3_3.set_ylabel('Pressure (bar)', color=color)  # we already handled the x-label with ax1
  ax3_3.tick_params(axis='y', labelcolor=color)
  ax3_3.spines['right'].set_position(('outward', 60))
  line3_2, = ax3_3.plot(thanos_contour.x, channelSim.Pco * 1e-5, color=color)

  #ax1.set_aspect('equal', adjustable='box')
  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  #ax3.set_yticks(np.arange(0, 3000, 500))
  ax[3].grid()
  df = pd.DataFrame(
      {
          "x": thanos_contour.x,
          "hg": thanos_thermals.hg1,
          "t_gas": thanos_thermals.T,
          "t_coolant": channelSim.Tco,
      }
  )
  df.to_excel('example_pandas.xlsx', index=False)
  plt.savefig("output")


aluminium = Metal(k=100, 
                  a=24e-6,
                  yield_array=np.array([
                    [298, 204e6],
                    [323, 198e6],
                    [373, 181e6],
                    [423, 182e6],
                    [473, 158e6],
                    [523, 132e6],
                    [573, 70e6],
                    [623, 30e6],
                    [673, 12e6]]),  #Linear interpolation of old_min, old_max, new_min, new_max. T1, T2, Stress1, Stress2
                  modulus_array=np.array([
                    [273,70e9], 
                    [773,50e9]])   #Linear interpolation of old_min, old_max, new_min, new_max. T1, T2, Stress1, Stress2
                  )

fuelmix_str = """
fuel CH3OH(L)   C 1 H 4 O 1
h,cal=-57040.0      t(k)=298.15       wt%=90.0
oxid water H 2 O 1  wt%=10.0
h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998
"""
add_new_fuel( 'fuelmix', fuelmix_str )



thanos = RocketEngine(
    oxName = "N2O",
    fuelName = "Methanol",
    thrust = 4000,
    Pc = 20,
    Pe = 0.85,
    MR = 2.5, #thats O/F
    radius_cylinder = 0.04)

from matplotlib import pyplot as plt

thanos_contour = Contour(thanos,
                         chamber_length=0.15,   #Chamber length
                         radius_cylinder=0.05,  #Chamber radius
                         points=100             #Discretized Points
                         )
thanos_thermals = Thermals(thanos, thanos_contour, 0,
                           hg_multiplier=1      #Funky wonky
                           )

test_hcmultilier = (thanos_contour.r / thanos_contour.throat[1])**0
thanos_channel = Channels(thanos, thanos_contour, thanos_thermals,
                          h=0.0015,   #Inner wall thickness
                          hc0=0.0015,  #Initial Channel Height
                          hcmultiplier = test_hcmultilier,  #Initial Channel Height
                          a0=0.004,   #Initial Channel Width
                          N=40        #Number of chan nels
                          )

channelSim = ChannelSim(thanos, thanos_contour, thanos_thermals, thanos_channel,
                        metal=aluminium   #Change material
                        )



displaysim(showtext=True)

