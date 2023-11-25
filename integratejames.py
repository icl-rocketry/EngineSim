from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
import numpy as np
import rocketcea.cea_obj
from matplotlib import pyplot as plt
from scipy.optimize import root_scalar
from pint import UnitRegistry
from pyfluids import Fluid, FluidsList, Input, Mixture
import pandas as pd
ureg = UnitRegistry()
Q_ = ureg.Quantity


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


def lininterp(x, old_min, old_max, new_min, new_max):
  return (x - old_min)/(old_max - old_min) * (new_max - new_min) + new_min

def pointinterp(x, start, end):
  return np.array([start[0] + x * (end[0] - start[0]), start[1] + x * (end[1] - start[1])])

class RocketEngine:
    def __init__(self, oxName, fuelName, thrust, Pc, Pe, MR, metal):
        self.oxName = oxName
        self.fuelName = fuelName
        self.Pc = Pc
        self.Pe = Pe
        self.thrust = thrust
        self.MR = MR
        self.metal = metal
    def defineGeometry(self, radius_cylinder, chamber_length, points, theta_n, theta_e, theta_c):
        self.radius_cylinder = radius_cylinder
        self.chamber_length = chamber_length
        self.points = points
        self.theta_n = theta_n
        self.theta_e = theta_e
        self.theta_c = theta_c
    def defineChannels(self, h, hc0, hcmultiplier, a0, N_channels, helical_angle):
        self.h = h
        self.hc0 = hc0
        self.hcmultiplier = hcmultiplier
        self.a0 = a0
        self.N_channels = N_channels
        self.helical_angle = helical_angle
    def runSim(self):
        # CEA
        self.cea = CEA_Obj(oxName = self.oxName,
                        fuelName = self.fuelName,
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

        self.eps = self.cea.get_eps_at_PcOvPe(Pc=self.Pc, MR=self.MR, PcOvPe = self.Pc/self.Pe)
        self.Tc, self.Tt, self.Te = self.cea.get_Temperatures(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.Pt = self.Pc / self.cea.get_Throat_PcOvPe(Pc=self.Pc, MR=self.MR)
        self.ve = self.cea.get_SonicVelocities(Pc=self.Pc,MR=self.MR,eps=self.eps)[2] * self.cea.get_MachNumber(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.mdot = self.thrust / self.ve
        self.mwt, self.gt = self.cea.get_Throat_MolWt_gamma(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.Rt = 8314.46 / self.mwt
        self.At = self.mdot / (self.Pt * 1e5) * np.sqrt(self.Rt * self.Tt / self.gt)
        self.CR = self.radius_cylinder**2 * np.pi / self.At
        self.cea = CEA_Obj(oxName = self.oxName,
                        fuelName = self.fuelName,
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

        self.eps = self.cea.get_eps_at_PcOvPe(Pc=self.Pc, MR=self.MR, PcOvPe = self.Pc/self.Pe)
        self.Tc, self.Tt, self.Te = self.cea.get_Temperatures(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.Pt = self.Pc / self.cea.get_Throat_PcOvPe(Pc=self.Pc, MR=self.MR)
        self.ve = self.cea.get_SonicVelocities(Pc=self.Pc,MR=self.MR,eps=self.eps)[2] * self.cea.get_MachNumber(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.mdot = self.thrust / self.ve

        self.mwt, self.gt = self.cea.get_Throat_MolWt_gamma(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.mwe, self.ge = self.cea.get_exit_MolWt_gamma(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.Rt = 8314.46 / self.mwt
        self.Re = 8314.46 / self.mwe

        self.Cp_c, self.Cp_t, self.Cp_e = self.cea.get_HeatCapacities(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.density_c = self.cea.get_Chamber_Density(Pc=self.Pc, MR=self.MR, eps=self.eps)

        _, self.visc, self.k, self.pr = self.cea.get_Chamber_Transport(Pc=self.Pc, MR=self.MR, eps=self.eps)
        self.cstar = self.cea.get_Cstar(Pc=self.Pc, MR=self.MR)
        self.isp = self.cea.estimate_Ambient_Isp(Pc=self.Pc, MR=self.MR, eps=self.eps, Pamb=1)

        self.At = self.mdot / (self.Pt * 1e5) * np.sqrt(self.Rt * self.Tt / self.gt)
        self.Ae = self.At * self.eps
        self.Ac = self.radius_cylinder**2 * np.pi
        self.CR = self.Ac / self.At

        # Contour
        rt = np.sqrt(self.At / np.pi)
        re = np.sqrt(self.Ae / np.pi)
        Le = 0.8 * rt * (np.sqrt(self.eps) - 1) / np.tan(15 * np.pi / 180)
        total_length = self.chamber_length + Le

        self.theta_n = 22 * np.pi / 180
        self.theta_e = 14 * np.pi / 180

        # Cyl (For my laziness assume R2 = 0)
        b = 35 * np.pi / 180
        x_conv = self.chamber_length + 1.5 * rt * np.cos(-b - np.pi / 2)
        y_conv = 2.5 * rt + 1.5 * rt * np.sin(-b - np.pi / 2)
        r_conv = (self.radius_cylinder - y_conv) / (1 - np.cos(b))
        y0 = self.radius_cylinder - r_conv
        x_slope = r_conv * np.sin(b)

        contour_cyl = np.array([
            [0, self.radius_cylinder],
            [x_conv - x_slope,self.radius_cylinder]])

        # Cylinder -> Converge
        contour_cylinder_conv = np.zeros((self.points - 1,2))
        for i in range(1, self.points):
            t = lininterp(i, 0, self.points - 1, 0, b)
            contour_cylinder_conv[i - 1] = np.array([x_conv - x_slope + r_conv * np.sin(t), y0 + r_conv * np.cos(t)])
            
        # Converge -> Throat
        contour_converge_throat = np.zeros((self.points - 1,2))
        for i in range(1, self.points):
            t = lininterp(i, 0, self.points - 1, -b - np.pi / 2, - np.pi / 2)
            contour_converge_throat[i - 1] = np.array([self.chamber_length + 1.5 * rt * np.cos(t), 2.5 * rt + 1.5 * rt * np.sin(t)])

        # Throat -> Parabola
            contour_throat_parabola = np.zeros((self.points - 1,2))
        for i in range(1, self.points):
            t = lininterp(i, 0, self.points - 1, 0 - np.pi / 2, self.theta_n - np.pi / 2)
            contour_throat_parabola[i - 1] = np.array([self.chamber_length + 0.382 * rt * np.cos(t), 1.382 * rt + 0.382 * rt * np.sin(t)])

        # Parabola
        parabola_p0 = contour_throat_parabola[-1]
        parabola_p2 = np.array([total_length, re])
        parabola_m1 = np.tan(self.theta_n)
        parabola_m2 = np.tan(self.theta_e)
        parabola_c1 = parabola_p0[1] - parabola_m1 * parabola_p0[0]
        parabola_c2 = parabola_p2[1] - parabola_m2 * parabola_p2[0]
        parabola_p1 = np.array([(parabola_c2 - parabola_c1)/(parabola_m1 - parabola_m2),(parabola_m1 * parabola_c2 - parabola_m2 * parabola_c1)/(parabola_m1 - parabola_m2)])

        contour_parabola = np.zeros((self.points - 1,2))
        for i in range(1, self.points):
            t = lininterp(i, 0, self.points - 1, 0, 1)
            parabola_linear0 = pointinterp(t, parabola_p0, parabola_p1)
            parabola_linear1 = pointinterp(t, parabola_p1, parabola_p2)
            contour_parabola[i - 1] = pointinterp(t, parabola_linear0, parabola_linear1)

        contour = np.concatenate([contour_cyl, contour_cylinder_conv, contour_converge_throat, contour_throat_parabola, contour_parabola])
        self.x = np.linspace(contour[:,0].min(), contour[:,0].max(), num=self.points)
        self.r = np.interp(self.x, contour[:,0], contour[:,1])

        self.throat = (self.chamber_length, rt)
        self.exit = (total_length, re)
        self.rc = self.radius_cylinder
        self.parabola_p1 = parabola_p1

        x_throat = np.argmin(self.r)
        self.x_throat = x_throat
        
        contours = np.split(contour,[x_throat])
        subar = contours[0]
        supar = contours[1]
        
        CEAString = self.cea.get_full_cea_output(Pc=self.Pc, MR=self.MR, short_output=1, pc_units="bar",subar=[3,2])
        DataArray = DataString.split()
        

        #Finding mach number, https://kyleniemeyer.github.io/gas-dynamics-notes/compressible-flows/isentropic.html#equation-eq-area-ratio-loss
        def area_function(mach, area, gamma):

            area_ratio = area / self.At
            if mach == 0:
                mach = 0.000001
            return (
                area_ratio - (
                    (1.0/mach) * ((1 + 0.5*(gamma-1)*mach*mach) /
                    ((gamma + 1)/2))**((gamma+1) / (2*(gamma-1)))
                    )
                )

        self.M = np.zeros(self.points)
        self.oldM = np.zeros(self.points)
        self.T = np.zeros(self.points)
        self.oldT = np.zeros(self.points)
        self.density = np.zeros(self.points)
        self.olddensity = np.zeros(self.points)
        self.hg1 = np.zeros(self.points)
        self.hg2 = np.zeros(self.points)
        self.hg3 = np.zeros(self.points)
        self.stag_recovery = np.zeros(self.points)
        self.gamma = np.zeros(self.points)
        self.oldgamma = np.zeros(self.points)
        #self.density = np.zeros(self.points)

        #Cv = self.Cp_c - self.Rt
        rt = np.sqrt(self.At / np.pi)

        for i in range(0,self.points):
            if i < x_throat:
                oldgamma = self.gt
            else:
                oldgamma = lininterp(i, x_throat, self.points, self.gt, self.ge)
                
            self.oldgamma[i] = oldgamma
                    
            area = np.pi * self.r[i] * self.r[i]
            if i == x_throat:
                oldM = 1
            elif i < x_throat:
                oldM = root_scalar(area_function, args=(area, oldgamma), bracket=[0, 1]).root
            else:
                oldM = root_scalar(area_function, args=(area, oldgamma), bracket=[1, 10]).root
            self.oldM[i] = oldM

            #T = self.Tc - self.mdot * self.mdot / (2 * Cv * self.density_c * self.density_c * area * area) * (1 + 0.5 * (self.gt - 1) * M * M)**(2 / (self.gt - 1))
            oldT = self.Tc / (1 + 0.5 * (oldgamma - 1) * oldM*oldM)
            self.oldT[i] = oldT
            olddensity = self.density_c * (1 + 0.5 * (oldgamma - 1) * oldM*oldM)**(-1/(oldgamma - 1))
            self.olddensity[i] = olddensity



def displaysim(showtext):
  if showtext:
    print(f'Mass flux (kg/s) = {thanos.mdot}')
    print(f'Throat      (mm) = {thanos.throat}')
    print(f'Exit        (mm) = {thanos.exit}')
    print(f'Cylinder r  (mm) = {thanos.rc}')
    print(f'Parabola    (mm) = {thanos.parabola_p1}')
    print(f'ISP         (mm) = {thanos.isp}')
    print(f'CR         (mm) = {thanos.CR}')

  fig, ax = plt.subplots(1, 1, sharey=True)

  color = 'tab:gray'
  ax.set_xlabel('position (m)')
  ax.set_ylabel('chamber radius (m)', color=color)
  ax.plot(thanos.x, thanos.r, color=color)
  ax.tick_params(axis='y', labelcolor=color)
  ax.set_ylim(0, 0.14)
  
  color = 'tab:blue'
  ax00_2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
  ax00_2.set_ylabel('Temperature (K)', color=color)  # we already handled the x-label with ax1
  ax00_2.plot(thanos.x, thanos.oldT, color=color, linestyle='dashed')
  #ax00_2.plot(thanos.x, thanos.T, color=color)
  ax00_2.spines['right'].set_position(('outward', 0))
  ax00_2.tick_params(axis='y', labelcolor=color)
  
  color = 'tab:orange'
  ax00_2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
  ax00_2.set_ylabel('Density (kg/m3)', color=color)  # we already handled the x-label with ax1
  ax00_2.plot(thanos.x, thanos.olddensity, color=color, linestyle='dashed')
  #ax00_2.plot(thanos.x, thanos.density, color=color)
  ax00_2.spines['right'].set_position(('outward', 60))
  ax00_2.tick_params(axis='y', labelcolor=color)
  
  color = 'tab:green'
  ax00_2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
  ax00_2.set_ylabel('Gamma', color=color)  # we already handled the x-label with ax1
  ax00_2.plot(thanos.x, thanos.oldgamma, color=color, linestyle='dashed')
  #ax00_2.plot(thanos.x, thanos.gamma, color=color)
  ax00_2.spines['right'].set_position(('outward', 120))
  ax00_2.tick_params(axis='y', labelcolor=color)

  fig.tight_layout()
  ax.grid()


thanos = RocketEngine(
    oxName = "N2O",
    fuelName = "Methanol",
    thrust = 4000,
    Pc = 20,
    Pe = 0.85,
    MR = 2.5, #thats O/F
    metal = aluminium)

thanos.defineGeometry(
    radius_cylinder=0.05,
    chamber_length=0.15,
    points=100,
    theta_n=22,
    theta_e=14,
    theta_c=35
)

thanos.defineChannels(
    h=0.0015,
    hc0=0.0015,
    hcmultiplier=np.ones(100),
    a0=0.004,
    N_channels=40,
    helical_angle=0
)
thanos.runSim()
displaysim(True)