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
        self.unitless_cea = rocketcea.cea_obj.CEA_Obj(oxName = self.oxName,fuelName = self.fuelName,fac_CR=self.CR,)

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

        _, self.oldvisc, self.oldk, self.oldpr = self.cea.get_Chamber_Transport(Pc=self.Pc, MR=self.MR, eps=self.eps)
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
        b = self.theta_c * np.pi / 180
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
        
        ar = self.r ** 2 * np.pi / self.At
        
        self.Lstar = sum(self.r[0:x_throat] ** 2 * np.pi * (self.x[1] - self.x[0])) / self.At
        
        self.P = np.zeros(self.points)
        self.T = np.zeros(self.points)
        self.mw = np.zeros(self.points)
        self.gamma = np.zeros(self.points)
        self.sonvel = np.zeros(self.points)
        self.M = np.zeros(self.points)
        self.pr = np.zeros(self.points)
        self.density = np.zeros(self.points)
        self.Cp = np.zeros(self.points)
        self.k = np.zeros(self.points)
        self.visc = np.zeros(self.points)
        

        self.hg1 = np.zeros(self.points)
        self.hg2 = np.zeros(self.points)
        self.stag_recovery = np.zeros(self.points)
        rt = np.sqrt(self.At / np.pi)


        def extractData(skip_index, lookfor_string, skip_times=0):
            j = 100
            for i in range(skip_times + 1):
                j = j + 1
                while CEAarray[j]!=lookfor_string:
                    j = j + 1;
            return CEAarray[j + skip_index]
        for i in range(self.points):

            r = self.r[i]
            area = r*r * np.pi


            if i <= x_throat:
                CEAstring = self.unitless_cea.get_full_cea_output(Pc=self.Pc, MR=self.MR, short_output=1, pc_units="bar",subar=ar[i], show_transport=1)
            else:
               CEAstring = self.unitless_cea.get_full_cea_output(Pc=self.Pc, MR=self.MR, short_output=1, pc_units="bar",eps=ar[i],show_transport=1)
            CEAarray = CEAstring.split()
            #print(CEAarray)
            P =         float(extractData(5, 'P,'))
            T =         float(extractData(5, 'T,'))
            mw =        float(extractData(5, 'M,'))
            gamma =     float(extractData(4, 'GAMMAs'))
            sonvel=     float(extractData(5, 'SON'))
            M =         float(extractData(5, 'MACH'))
            pr =        float(extractData(5, 'PRANDTL'))
            density =   float(extractData(5, 'RHO,').replace('-','e-')) * 1000
            Cp =        float(extractData(5, 'Cp,')) * 4186.8
            k =         float(extractData(4, 'CONDUCTIVITY', 1)) * 0.4184
            visc =      float(extractData(4, 'VISC,MILLIPOISE')) * 0.0001

            self.P[i] = P
            self.T[i] = T
            self.mw[i] = mw
            self.gamma[i] = gamma
            self.sonvel[i] = sonvel
            self.M[i] = M
            self.pr[i] = pr
            self.density[i] = density
            self.Cp[i] = Cp
            self.k[i] = k
            self.visc[i] = visc

            correction_factor = 1 / ((0.5 * T / self.Tc * (1 + 0.5 * (gamma - 1) * M**2)+0.5)**0.68 * (1 + 0.5 * (gamma - 1) * M**2)**0.12)


            self.stag_recovery[i] = (1 + 0.5*(gamma - 1)*M**2*pr**0.33)

            #Bartz
            bmPc = self.Pc * 1e5
            bmcstar = self.cstar
            bmDt = np.sqrt(4 * self.At / np.pi)
            bmcurv = (1.5 * rt + 0.382 * rt) * 0.5
            bmCp = Cp
            bmvisc = visc
            

            bPc = Q_(self.Pc, ureg.bar).to('psi')
            bcstar = Q_(self.cstar, ureg.meter / ureg.second).to('feet / second')
            bDt = Q_(np.sqrt(4 * self.At / np.pi), ureg.meter).to('inch')
            bcurv = Q_((1.5 * rt + 0.382 * rt) * 0.5, ureg.meter).to("inch")
            bCp = Q_(Cp, ureg.joule / (ureg.kilogram * ureg.degK)).to('Btu / (lb * degR)')
            bvisc = Q_(visc, ureg.pascal * ureg.second).to('lbf * second / inch**2')
            bg = Q_(9.81, ureg.meter / ureg.second**2).to('feet / second**2')
            bAtA = self.At / area
    
            
            
            
            self.hg1[i] = 0.026 / bmDt**0.2 * (bmvisc**0.2 * bmCp / pr**0.6) * (bmPc / bmcstar)**0.8 * (bmDt / bmcurv)**0.1 * bAtA**0.9 * correction_factor
            
            R = self.Rt
            Z = np.pi * self.r[0]**2 / (2 * np.pi * self.r[0] * self.x[-1])
            self.hg2[i] = Z * self.mdot / (2 * area) * Cp * visc**0.3 * pr **(2/3)
            
        
            self.hc = np.zeros(self.points)
        self.a = np.zeros(self.points)
        self.A = np.zeros(self.points)
        self.per = np.zeros(self.points)

        r0 = self.r[0]

        for i, r in enumerate(self.r):
            hc = self.hc0 * self.hcmultiplier[i]
            a = self.a0 * (r + self.h) / (r0 + self.h)
            a2 = self.a0 * (r + self.h + hc) / (r0 + self.h)

            self.hc[i] = hc
            self.a[i] = a
            self.A[i] = hc * a
            self.per[i] = 2 * hc + a + a2


        #----------------------------------------------------------------------
        mdot = self.mdot / (self.N_channels * (self.MR + 1))
        self.Taw = np.zeros(self.points)
        self.Twg = np.zeros(self.points)
        self.Twc = np.zeros(self.points)
        self.Tco = np.zeros(self.points)
        self.velocity = np.zeros(self.points)
        self.hg = np.zeros(self.points)
        self.hc_thermal = np.zeros(self.points)
        self.hw = np.zeros(self.points)
        self.q = np.zeros(self.points)
        self.stress_pressure = np.zeros(self.points)
        self.stress_temperature = np.zeros(self.points)
        self.stress_temperature2 = np.zeros(self.points)
        self.stress_total = np.zeros(self.points)
        self.Pco = np.zeros(self.points)
        self.viscosity_co = np.zeros(self.points)
        self.density_co = np.zeros(self.points)        
        Tco_i = 300
        Pco_i = 40e5

        for i in range(self.points-1, -1, -1):
            r = self.r[i]
            T_inf = self.T[i]
            stag_recovery = self.stag_recovery[i]
            Taw = T_inf * 0.923
            hg = self.hg2[i]

            a = self.a[i]
            A = self.A[i]
            per = self.per[i]

            
            #coolant = Fluid(FluidsList.Methanol).with_state(Input.pressure(Pco_i), Input.temperature(Tco_i-275.15))
            coolant = Mixture([FluidsList.Water, FluidsList.Methanol], [10, 90]).with_state(Input.pressure(Pco_i), Input.temperature(Tco_i-275.15))

            k = coolant.conductivity
            density = coolant.density
            pr = coolant.prandtl
            viscosity = coolant.dynamic_viscosity
            cp = coolant.specific_heat
            
            self.viscosity_co[i] = viscosity
            self.density_co[i] = density

            Dh = 4 * A / per
            velocity = mdot/(A * density * np.cos(self.helical_angle * np.pi / 180))
            #velocity = mdot/(A * density * np.sin(np.arctan(204 / (75 * np.pi) * 0.075 / r)))
            #velocity = mdot/(A * density)

            Twc = Tco_i + 100
            for j in range(0, 10):
                hc = 0.023 * k / Dh * (density * velocity * Dh / viscosity)**0.8 * pr ** 0.4 * (Twc / Tco_i)**-0.3
                self.hc_thermal[i] = hc
                H = 1 / (1 / hg + self.h / self.metal.k + 1 / hc)
                #H = 1 / (1 / hg + 0.00005/1 + self.h / self.metal.k + 1 / hc)
                q = H * (Taw - Tco_i)
                Twc = Tco_i + q / hc

            Twg = Twc + q * self.h / self.metal.k

            stress_pressure = 0.5 * (40e5 - 1e5) * (a / self.h) **2;
            stress_temperature = self.metal.modulus(Twg) * self.metal.a * q * self.h * 0.5 / ((1 - 0.3) * self.metal.k)
            stress_temperature2 = self.metal.modulus(Twg) * self.metal.a * (Twg - Twc)

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
            #self.hc[i] = hc
            self.hw[i] = self.metal.k / self.h
            self.q[i] = q
            self.stress_pressure[i] = stress_pressure
            self.stress_temperature[i] = stress_temperature
            self.stress_temperature2[i] = stress_temperature2
            self.stress_total[i] = stress_pressure + stress_temperature + stress_temperature2
            self.Pco[i] = Pco_i

            dx = self.x[1] - self.x[0]
            Tco_i = Tco_i + 1 / (mdot * cp) * q * (2 * np.pi * r * dx / self.N_channels)
            #Pco_i = Pco_i - 32 * cf * dx * mdot **2 / (density * np.pi**2 * Dh**5)
            Pco_i = Pco_i - 4 * cf * dx / Dh * density * velocity ** 2


def displaysim(showtext):
  if showtext:
    print(f'Mass flux (kg/s) = {thanos.mdot}')
    print(f'Throat      (mm) = {thanos.throat}')
    print(f'Exit        (mm) = {thanos.exit}')
    print(f'Cylinder r  (mm) = {thanos.rc}')
    print(f'Parabola    (mm) = {thanos.parabola_p1}')
    print(f'ISP         (mm) = {thanos.isp}')
    print(f'CR         (mm) = {thanos.CR}')

  fig, ax = plt.subplots(2, 3, figsize=(15, 7),sharey=True)

  color = 'tab:gray'
  ax[0,0].set_xlabel('position (m)')
  ax[0,0].set_ylabel('chamber radius (m)', color=color)
  ax[0,0].plot(thanos.x, thanos.r, color=color)
  ax[0,0].tick_params(axis='y', labelcolor=color)
  ax[0,0].set_ylim(0, 0.14)

  color = 'tab:blue'
  ax00_2 = ax[0,0].twinx()  # instantiate a second axes that shares the same x-axis
  ax00_2.set_ylabel('temperature (K)', color=color)  # we already handled the x-label with ax1
  ax00_2.plot(thanos.x, thanos.T, color=color)
  ax00_2.spines['right'].set_position(('outward', 60))
  ax00_2.tick_params(axis='y', labelcolor=color)

  color = 'tab:orange'
  ax00_3 = ax[0,0].twinx()  # instantiate a second axes that shares the same x-axis
  ax00_3.set_ylabel('hg', color=color)  # we already handled the x-label with ax1
  line00_1, = ax00_3.plot(thanos.x, thanos.hg1, color=color, label='Bartz')
  line00_2, = ax00_3.plot(thanos.x, thanos.hg2, color=color, label='Adami')
  ax00_3.tick_params(axis='y', labelcolor=color)
  #ax00_3.set_yticks(np.arange(0, 3000, 500))
  fig.tight_layout()
  ax[0,0].grid()
  ax00_3.legend(handles=[line00_1, line00_2], loc='upper right')
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
  color = 'tab:gray'
  ax[0,1].set_xlabel('position (m)')
  ax[0,1].set_ylabel('chamber radius (m)', color=color)
  ax[0,1].plot(thanos.x, thanos.r, color=color)
  #ax[1].plot(thanos.x, thanos.r + thanos_channel.h, color=color)
  #ax[1].plot(thanos.x, thanos.r + thanos_channel.h + thanos_channel.hc, color=color)
  ax[0,1].tick_params(axis='y', labelcolor=color)
  #ax[0,1].set_box_aspect(1)

  color = 'tab:red'
  ax01_2 = ax[0,1].twinx()  # instantiate a second axes that shares the same x-axis
  ax01_2.set_ylabel('Hot Gas Wall Temperature (K)', color=color)  # we already handled the x-label with ax1
  ax01_2.tick_params(axis='y', labelcolor=color)
  line01_1, = ax01_2.plot(thanos.x, thanos.Twg, color='tab:red', label='T Combustion Side Wall')
  line01_2, = ax01_2.plot(thanos.x, thanos.Twc, color='tab:orange', label='T Coolant Side Wall')
  line01_3, = ax01_2.plot(thanos.x, thanos.Tco, color='tab:blue', label='T Coolant Bulk')
  ax[0,1].grid()
  ax01_2.legend(handles=[line01_1, line01_2, line01_3], loc='upper left')
#----------------------------------------------------------------------------------------------------------------------------------------------------------------
  color = 'tab:gray'
  ax[0,2].set_xlabel('position (m)')
  ax[0,2].set_ylabel('chamber radius (m)', color=color)
  ax[0,2].plot(thanos.x, thanos.r, color=color)
  ax[0,2].tick_params(axis='y', labelcolor=color)
  ax02_2 = ax[0,2].twinx()  # instantiate a second axes that shares the same x-axis
  color = 'tab:orange'
  ax02_2.set_ylabel('Stress (MPa)', color=color)  # we already handled the x-label with ax1
  ax02_2.tick_params(axis='y', labelcolor=color)
  line02_1, = ax02_2.plot(thanos.x, thanos.stress_total * 1e-6, color='tab:pink', label='Total Stress')
  line02_3, = ax02_2.plot(thanos.x, thanos.stress_pressure * 1e-6, color='tab:purple', label='Pressure Stress')
  line02_2, = ax02_2.plot(thanos.x, thanos.metal.yield_stress(thanos.Twg) * 1e-6, color='tab:green', label='Yield Stress')

  #ax1.set_aspect('equal', adjustable='box')
  #fig.tight_layout()  # otherwise the right y-label is slightly clipped
  #ax3.set_yticks(np.arange(0, 3000, 500))
  ax[0,2].grid()
  ax02_2.legend(handles=[line02_1, line02_2, line02_3], loc='upper left')
  #----------------------------------------------------------------------------------------------------------------------------------------------------------------
  color = 'tab:gray'
  ax[1,0].set_xlabel('position (m)')
  ax[1,0].set_ylabel('chamber radius (m)', color=color)
  ax[1,0].plot(thanos.x, thanos.r, color=color)
  ax[1,0].tick_params(axis='y', labelcolor=color)
  ax[1,0].set_ylim(0, 0.14)

  color='tab:orange'
  ax10_2 = ax[1,0].twinx()  # instantiate a second axes that shares the same x-axis
  ax10_2.set_ylabel('Velocity (m/d)', color=color)  # we already handled the x-label with ax1
  ax10_2.tick_params(axis='y', labelcolor=color)
  line10_1, = ax10_2.plot(thanos.x, thanos.velocity, color=color)

  color='tab:pink'
  ax10_3 = ax[1,0].twinx()  # instantiate a second axes that shares the same x-axis
  ax10_3.set_ylabel('Pressure (bar)', color=color)  # we already handled the x-label with ax1
  ax10_3.tick_params(axis='y', labelcolor=color)
  ax10_3.spines['right'].set_position(('outward', 60))
  line01_2, = ax10_3.plot(thanos.x, thanos.Pco * 1e-5, color=color)

  #ax1.set_aspect('equal', adjustable='box')
  #fig.tight_layout()  # otherwise the right y-label is slightly clipped
  #ax3.set_yticks(np.arange(0, 3000, 500))
  ax[1,0].grid()
  plt.savefig("output")


def displaysim2(showtext):
  if showtext:
    print(f'Mass flux (kg/s) = {thanos.mdot}')
    print(f'Throat      (mm) = {thanos.throat}')
    print(f'Exit        (mm) = {thanos.exit}')
    print(f'Cylinder r  (mm) = {thanos.rc}')
    print(f'Parabola    (mm) = {thanos.parabola_p1}')
    print(f'ISP         (mm) = {thanos.isp}')
    print(f'CR          (mm) = {thanos.CR}')
    print(f'L*          (mm) = {thanos.Lstar}')

  fig, ax = plt.subplots(1, 1, sharey=True, figsize=[10,7])

  color = 'tab:gray'
  ax.set_xlabel('position (m)')
  ax.set_ylabel('chamber radius (m)', color=color)
  ax.plot(thanos.x, thanos.r, color=color)
  ax.tick_params(axis='y', labelcolor=color)
  ax.set_ylim(0, 0.14)
  
  color = 'tab:red'
  ax00_2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
  ax00_2.set_ylabel('Temperature', color=color)  # we already handled the x-label with ax1
  ax00_2.plot(thanos.x, thanos.T, color=color)
  ax00_2.spines['right'].set_position(('outward', 0))
  ax00_2.tick_params(axis='y', labelcolor=color)
  
  color = 'tab:blue'
  ax00_2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
  ax00_2.set_ylabel('Mach Number', color=color)  # we already handled the x-label with ax1
  ax00_2.plot(thanos.x, thanos.M, color=color)
  ax00_2.spines['right'].set_position(('outward', 60))
  ax00_2.tick_params(axis='y', labelcolor=color)
  
  color = 'tab:green'
  ax00_2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
  ax00_2.set_ylabel('Viscosity', color=color)  # we already handled the x-label with ax1
  ax00_2.plot(thanos.x, thanos.visc , color=color)
  ax00_2.spines['right'].set_position(('outward', 120))
  ax00_2.tick_params(axis='y', labelcolor=color)

  #fig.tight_layout()
  ax.grid()


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
    thrust = 1500,
    Pc = 20,
    Pe = 1,
    MR = 2.2, #thats O/F
    metal = aluminium)

thanos.defineGeometry(
    radius_cylinder=0.046,
    chamber_length=0.15,
    points=100,
    theta_n=22,
    theta_e=14,
    theta_c=35
)

thanos.defineChannels(
    h=0.008,
    hc0=0.0015,
    hcmultiplier=np.ones(100),
    a0=0.003,
    N_channels=40,
    helical_angle=0
)
thanos.runSim()
displaysim(True)

hc = thanos.hc
a = thanos.a
x = thanos.x
r = thanos.r
h = thanos.h
np.save('h',h)
np.save('r',r)
np.save('z',x)
np.save('hc',hc)
np.save('a',a)

#thanos.fuelName = "Ethanol"
#thanos.runSim()
#displaysim(True)

# '''
# fig, ax = plt.subplots(1, 1)
# I = np.linspace(0, 40, 10)
# OFLIST = np.linspace(2.5,2.5,1)
# yieldstress = np.zeros(10)
# totalstress = np.zeros(10)
# for j, o_f in enumerate(OFLIST):
#     thanos.MR = o_f
#     for i, h in enumerate(I):
#         thanos.helical_angle = h
#         thanos.runSim()
#         yieldstress[i] = thanos.metal.yield_stress(thanos.Twg[thanos.x_throat])
#         totalstress[i] = thanos.stress_total[thanos.x_throat]
#     plt.plot(I, yieldstress * 1e-6 - totalstress * 1e-6, color='green', label=str(o_f))
#     #plt.plot(I, totalstress * 1e-6, color='blue', label='Max stress at throat')
# plt.xlabel('Helical angle')
# plt.ylabel('Stress (MPa)')
# plt.legend()
# plt.show()
# fig, ax = plt.subplots(1, 1)
# I = np.linspace(0.0001, 0.002, 10)
# yieldstress = np.zeros(10)
# totalstress = np.zeros(10)
# for i, value in enumerate(I):
#     thanos.h = value
#     thanos.runSim()
#     yieldstress[i] = thanos.metal.yield_stress(thanos.Twg[thanos.x_throat])
#     totalstress[i] = thanos.stress_total[thanos.x_throat]
# plt.plot(I, yieldstress * 1e-6 - totalstress * 1e-6, color='green')
#     #plt.plot(I, totalstress * 1e-6, color='blue', label='Max stress at throat')
# plt.xlabel('Wall thickness')
# plt.ylabel('Stress (MPa)')
# plt.legend()
# plt.show()

# fig, ax = plt.subplots(1, 2, figsize=[20, 7])
# I = np.linspace(2, 4, 10)
# yieldstress = np.zeros(10)
# totalstress = np.zeros(10)
# isp = np.zeros(10)
# for i, value in enumerate(I):
#     thanos.MR = value
#     thanos.runSim()
#     yieldstress[i] = thanos.metal.yield_stress(thanos.Twg[thanos.x_throat])
#     totalstress[i] = thanos.stress_total[thanos.x_throat]
#     isp[i] = thanos.isp[0]
# ax[0].plot(I, yieldstress * 1e-6 - totalstress * 1e-6, color='green')
# ax[1].plot(I, isp, color='blue')

#     #plt.plot(I, totalstress * 1e-6, color='blue', label='Max stress at throat')
# ax[0].set_xlabel('O/F')
# ax[0].set_ylabel('Stress (MPa)')
# ax[1].set_xlabel('O/F')
# ax[1].set_ylabel('ISP (seconds)')
# plt.legend()
# plt.show()