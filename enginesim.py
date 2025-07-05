from rocketcea.cea_obj_w_units import CEA_Obj
from rocketcea.cea_obj import add_new_fuel
from pyfluids import Fluid, FluidsList, Input
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from os import path, system
import numpy as np
import scipy as sp
import csv
import warnings

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

g = sp.constants.g

# with open('film_stability.csv') as data: # [Re, eta]
#     film_stab_coeff = []
#     film_stab_re = []
#     filtered_data = csv.reader(data)
#     for line in filtered_data:
#         film_stab_re.append(float(line[0]))
#         film_stab_coeff.append(float(line[1]))

in2mm = lambda x: x * 25.4
mm2in = lambda x: x / 25.4
psi2bar = lambda x: x / 14.5038
bar2psi = lambda x: x * 14.5038

def OFsweep(fuel, ox, OFstart, OFend, pc, pe = None, cr = None, pamb=1.01325, showvacisp=False, filmcooled=False, film_perc=0):
        __doc__ = """
            Performs a sweep analysis of the engine performance across a range of OF ratios and plots the ISP and chamber temperature.
            
            Parameters
            ----------
            fuel : str
                Fuel name (e.g., 'Isopropanol', 'Ethanol')
            ox : str
                Oxidizer name (e.g., 'N2O', 'LOX')
            OFstart : float
                Starting value for OF ratio
            OFend : float
                Ending value for OF ratio
            pc : float
                Chamber pressure in bar
            pe : float
                Exit pressure in bar
            pamb : float
                Ambient pressure in bar (default: 1.01325)
            cr : float
                Contraction ratio
            showvacisp : bool, optional
                Whether to display vacuum ISP on the plot (default: False)
            filmcooled : bool, optional
                Whether film cooling is being used (default: False)
            film_perc : float, optional
                Film cooling percentage (default: 0)
                
            Returns
            -------
            None
                Creates and displays a plot with OF ratio on x-axis,
                ISP values on primary y-axis and chamber temperature on secondary y-axis.
                
            Notes
            -----
            This method dynamically adjusts the expansion ratio (eps) for each OF value
            to maintain the specified pressure ratio between chamber and exit.
            If filmcooled=True, it also calculates and plots the corrected ISP accounting
            for the performance loss due to film cooling.
        """
        film_frac = film_perc * 1e-2

        if pe is None:
            pe = pamb

        if cr is None:
            cr_string = 'CR = inf'
            ceaObj = CEA_Obj(
                oxName=ox,
                fuelName=fuel,
                isp_units='sec',
                cstar_units = 'm/s',
                pressure_units='Bar',
                temperature_units='K',
                sonic_velocity_units='m/s',
                enthalpy_units='J/g',
                density_units='kg/m^3',
                specific_heat_units='J/kg-K',
                viscosity_units='centipoise', # stored value in pa-s
                thermal_cond_units='W/cm-degC', # stored value in W/m-K
                make_debug_prints=False)
        else:
            cr_string = f'CR = {cr:.2f}'
            ceaObj = CEA_Obj(
                oxName=ox,
                fuelName=fuel,
                isp_units='sec',
                cstar_units = 'm/s',
                pressure_units='Bar',
                temperature_units='K',
                sonic_velocity_units='m/s',
                enthalpy_units='J/g',
                density_units='kg/m^3',
                specific_heat_units='J/kg-K',
                viscosity_units='centipoise', # stored value in pa-s
                thermal_cond_units='W/cm-degC', # stored value in W/m-K
                fac_CR=cr,
                make_debug_prints=False)
        
        OFs = np.linspace(OFstart,OFend,100)
        ispseas = []
        ispseas_true = []
        ispvacs = []
        Tcs = []
        for OF in OFs:
            eps = ceaObj.get_eps_at_PcOvPe(Pc=pc, MR=OF, PcOvPe=(pc/pe))
            [ispsea, _] = ceaObj.estimate_Ambient_Isp(Pc=pc, MR=OF, eps=eps, Pamb=pamb, frozen=0, frozenAtThroat=0)
            if filmcooled == True:
                ispsea_true = ispsea / (1 + (film_frac * (1/(OF + 1))))
                ispseas_true.append(ispsea_true)
            [ispvac, _, Tc] = ceaObj.get_IvacCstrTc(Pc=pc, MR=OF, eps=eps, frozen=0, frozenAtThroat=0)
            ispseas.append(ispsea)
            ispvacs.append(ispvac)
            Tcs.append(Tc)

        # Find peak values and their corresponding OFs
        peak_sl_isp_idx = np.argmax(ispseas)
        peak_sl_isp_OF = OFs[peak_sl_isp_idx]
        peak_vac_isp_idx = np.argmax(ispvacs)
        peak_vac_isp_OF = OFs[peak_vac_isp_idx]
        peak_tc_idx = np.argmax(Tcs)
        peak_tc_OF = OFs[peak_tc_idx]

        fig, ax1 = plt.subplots(figsize=(16, 9))
        ax1.set_xlabel('OF Ratio')
        ax1.set_ylabel('ISP (s)', color='b')
        ax1.plot(OFs, ispseas, 'b', label='SL ISP')
        if showvacisp == True:
            ax1.plot(OFs, ispvacs, 'b--', label='Vac ISP')
        if filmcooled == True:
            ax1.plot(OFs, ispseas_true, 'b--', label='True SL ISP')
        ax2 = ax1.twinx()
        ax2.plot(OFs, Tcs, 'r',label='Chamber Temp')
        ax2.set_ylabel('Chamber Temp (K)', color='r',)
        ax1.grid()
        ax1.grid(which="minor", alpha=0.5)
        ax1.minorticks_on()
        ax1.set_ylim(bottom=0)
        plt.xlim(OFstart, OFend)

        height = 0.8
        ax1.annotate(f"Peak SL ISP: {ispseas[peak_sl_isp_idx]:.1f}s, OF={peak_sl_isp_OF:.2f}",
            xy=(peak_sl_isp_OF, ispseas[peak_sl_isp_idx]),
            xytext=(peak_sl_isp_OF, ispseas[peak_sl_isp_idx] * height),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
            ha='center')
        ax1.plot([peak_sl_isp_OF, peak_sl_isp_OF], 
             [ispseas[peak_sl_isp_idx] * height, ispseas[peak_sl_isp_idx]], 
             'b-', linewidth=0.8)

        if showvacisp == True:
            height = 0.9
            ax1.annotate(f"Peak Vac ISP: {ispvacs[peak_vac_isp_idx]:.1f}s, OF={peak_vac_isp_OF:.2f}",
            xy=(peak_vac_isp_OF, ispvacs[peak_vac_isp_idx]),
            xytext=(peak_vac_isp_OF, ispvacs[peak_vac_isp_idx] * height),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
            ha='center')
            ax1.plot([peak_vac_isp_OF, peak_vac_isp_OF], 
                 [ispvacs[peak_vac_isp_idx] * height, ispvacs[peak_vac_isp_idx]], 
                 'b-', linewidth=0.8)
        
        height = 0.7
        ax2.annotate(f"Peak Tc: {Tcs[peak_tc_idx]:.0f}K, OF={peak_tc_OF:.2f}",
            xy=(peak_tc_OF, Tcs[peak_tc_idx]),
            xytext=(peak_tc_OF, Tcs[peak_tc_idx] * height),
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8),
            ha='center')
        # Add a line instead of an arrow
        ax2.plot([peak_tc_OF, peak_tc_OF], 
             [Tcs[peak_tc_idx] * height, Tcs[peak_tc_idx]], 
             'r-', linewidth=0.8)
        
        # Create a combined legend for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        
        # Only create legend if there are labeled items
        if lines:
            ax1.legend(lines, labels, loc='best')
        plt.title(f'OF Sweep\n{fuel} / {ox}, Pc = {pc:.2f} bar, Pe = {pe:.2f} bar, Pamb = {pamb:.2f} bar, {cr_string}')
        fig.tight_layout()

class engine:
    def __init__(self, file):
        self.file = file
        if path.exists(self.file) and path.getsize(self.file) > 0:
            with open(self.file, 'r') as input_file:
                data = input_file.readlines()
                is_rao = data[0].split()[-1]
                self.dc = float(data[1].split()[-1]) * 1e-3
                self.dt = float(data[2].split()[-1]) * 1e-3
                self.de = float(data[3].split()[-1]) * 1e-3
                self.lc = float(data[4].split()[-1]) * 1e-3
                self.R2 = float(data[5].split()[-1]) * 1e-3
                if is_rao.lower() == 'true':
                    self.rao = True
                    self.theta_n = float(data[7].split()[-1])
                    self.theta_e = float(data[8].split()[-1])
                    self.le = float(data[9].split()[-1]) * 1e-3
                    self.ln = 0
                elif is_rao.lower() == 'false':
                    self.rao = False
                    self.ln = float(data[8].split()[-1]) * 1e-3
                    self.div_angle = float(data[7].split()[-1])
                else:
                    raise ValueError("Invalid value for 'is_rao_nozzle' in config file. Expected 'True' or 'False'.")
                self.conv_angle = float(data[6].split()[-1])
                
            self.eps = (self.de/self.dt)**2
            self.rc = self.dc/2
            self.rt = self.dt/2
            self.re = self.de/2
            self.ac = np.pi*self.rc**2
            self.at = np.pi*self.rt**2
            self.ae = np.pi*self.re**2
            self.cr = self.ac/self.at
            if self.rao:
                self.rao_percentage = 100 * np.tan(np.deg2rad(15)) * self.le / ((np.sqrt(self.eps)-1)*self.rt)
            else:
                self.le = (self.re - self.rt) / np.tan(np.deg2rad(self.div_angle))
            self.R2_max = (self.rc - self.rt)/(1 - np.cos(np.deg2rad(self.conv_angle))) - 1.5*self.rt
            if self.R2 == -1:
                self.R2 = self.R2_max
            self.ltotal = self.lc + self.le + self.ln
            self.cstar_eff = 1

    def update(self):
        self.rc = self.dc/2
        self.rt = self.dt/2
        self.re = self.de/2
        self.ac = np.pi*self.rc**2
        self.at = np.pi*self.rt**2
        self.ae = np.pi*self.re**2
        self.cr = self.ac/self.at      
        self.eps = (self.de/self.dt)**2
        if self.rao:
            self.rao_percentage = 100 * np.tan(np.deg2rad(15)) * self.le / ((np.sqrt(self.eps)-1)*self.rt)
        else:
            self.le = (self.re - self.rt) / np.tan(np.deg2rad(self.div_angle))
        self.ltotal = self.lc + self.le + self.ln

    def gen_cea_obj(self):
        self.cea = CEA_Obj(
            oxName = self.ox,
            fuelName = self.fuel,
            isp_units='sec',
            cstar_units = 'm/s',
            pressure_units='Bar',
            temperature_units='K',
            sonic_velocity_units='m/s',
            enthalpy_units='J/g',
            density_units='kg/m^3',
            specific_heat_units='J/kg-K',
            viscosity_units='centipoise', # stored value in pa-s
            thermal_cond_units='W/cm-degC', # stored value in W/m-K
            fac_CR=self.cr,
            make_debug_prints=False)

    def combustion_sim(self, fuel, ox, OF, pc, pamb = 1.01325, cstar_eff = 1, sizing=False, **kwargs):
        __doc__ = """Simulates combustion of the engine and optionally sizes the engine.
            Parameters:
            -----------
            fuel : str
                Fuel propellant name
            ox : str 
                Oxidizer propellant name
            OF : float
                OF Ratio
            pc : float
                Chamber pressure in bar
            pamb : float, optional
                Ambient pressure in bar, default is 1.01325 (sea level)
            cstar_eff : float, optional
                Characteristic velocity efficiency, default is 1.0 (100%)
            sizing : bool, optional
                If True, performs engine sizing calculations, default is False
            **kwargs : dict
                Additional parameters required if sizing=True:
                - thrust : float
                    Desired thrust in Newtons
                - pe : float
                    Exit pressure in bar
                - cr : float
                    Contraction ratio (Ac/At)
                - conv_angle : float
                    Convergent angle in degrees
                - lstar : float
                    Characteristic chamber length in meters
                - rao_percentage : float
                    Percentage of the bell nozzle (Rao approximation)
            Returns:
            --------
            None
                Updates various engine performance attributes:
                - Performance: isp, cstar, cf, thrust, etc.
                - Mass Flow rates: mdot, ox_mdot, fuel_mdot
                - Gas properties: temperatures, pressures, transport properties
                - If sizing=True: engine dimensions and contour geometry
            Raises:
            -------
            ValueError
                If L* is too short or contraction ratio is too high during sizing
            Notes:
            ------
            This method uses CEA (Chemical Equilibrium with Applications) to simulate
            engine performance. When sizing=True, it also generates the engine geometry
            including chamber and nozzle contours.
            """
        self.fuel = fuel
        self.ox = ox
        self.OF = OF
        self.pc = pc
        self.pamb = pamb
        self.cstar_eff = cstar_eff

        if sizing == True:
            self.thrust = kwargs['thrust']
            self.pe = kwargs['pe']
            self.cr = kwargs['cr']
            self.conv_angle = kwargs['conv_angle']
            self.lstar = kwargs['lstar']
            self.rao_percentage = kwargs['rao_percentage']

        self.gen_cea_obj()

        if sizing == True:
            self.eps = self.cea.get_eps_at_PcOvPe(Pc=self.pc, MR=self.OF, PcOvPe=(self.pc/self.pe))

        [self.ispvac, self.cstar, _] = self.cea.get_IvacCstrTc(Pc=self.pc, MR=self.OF, eps=self.eps)
        [self.ispsea, _] = self.cea.estimate_Ambient_Isp(Pc=self.pc, MR=self.OF, eps=self.eps, Pamb=1.01325, frozen=0, frozenAtThroat=0)
        [self.Tg_c, self.Tg_t, self.Tg_e] = self.cea.get_Temperatures(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=0, frozenAtThroat=0)
        self.ispsea = self.ispsea * self.cstar_eff
        self.ispvac = self.ispvac * self.cstar_eff
        self.cstar = self.cstar * self.cstar_eff
        self.pt = self.pc/self.cea.get_Throat_PcOvPe(Pc=self.pc, MR=self.OF)
        [_, self.cf, self.exitcond] = self.cea.get_PambCf(Pamb=self.pamb, Pc=self.pc, MR=self.OF, eps=self.eps)
        if sizing == True:
            self.at = self.thrust / (self.pc * self.cf * 1e5)
        self.mdot = self.pc * 1e5 * self.at / self.cstar
        self.ox_mdot = self.mdot * self.OF / (1 + self.OF)
        self.fuel_mdot = self.mdot / (1 + self.OF)
        if sizing == False:
            self.thrust = self.cf * self.mdot * self.cstar
        self.isp = self.thrust / (self.mdot * g)
        self.PinjPcomb = self.cea.get_Pinj_over_Pcomb(Pc=self.pc, MR=self.OF)
        self.Me = self.cea.get_MachNumber(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=0, frozenAtThroat=0)
        [self.cp_c, self.mu_c, self.k_c, self.pr_c] = self.cea.get_Chamber_Transport(Pc=self.pc, MR=self.OF, eps=self.eps)
        [self.cp_t, self.mu_t, self.k_t, self.pr_t] = self.cea.get_Throat_Transport(Pc=self.pc, MR=self.OF, eps=self.eps)
        [self.cp_e, self.mu_e, self.k_e, self.pr_e] = self.cea.get_Exit_Transport(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=0, frozenAtThroat=0)
        [_, self.gam_t] = self.cea.get_Throat_MolWt_gamma(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=0)
        [_, self.gam_c] = self.cea.get_Chamber_MolWt_gamma(Pc=self.pc, MR=self.OF, eps=self.eps)
        [_, self.gam_e] = self.cea.get_exit_MolWt_gamma(Pc=self.pc, MR=self.OF, eps=self.eps, frozen=0, frozenAtThroat=0)
        self.mu_c = self.mu_c * 1e-3 # now pa-s
        self.k_c = self.k_c * 100 # now W/m-K

        if sizing == True:
            self.rt = np.sqrt(self.at/np.pi)
            self.dt = 2*self.rt

            self.ae = self.at * self.eps
            self.re = np.sqrt(self.ae/np.pi)
            self.de = 2*self.re

            self.ac = self.at * self.cr
            self.rc = np.sqrt(self.ac/np.pi)
            self.dc = 2*self.rc

            self.R2 = (self.rc - self.rt)/(1 - np.cos(np.deg2rad(self.conv_angle))) - 1.5*self.rt

            self.n_points = 100
            l = np.linspace(0, (self.R2*np.sin(np.deg2rad(self.conv_angle))), self.n_points)
            r = np.sqrt(self.R2**2 - l**2) + self.rc - self.R2
            l2 = np.linspace(0, (1.5*self.rt*np.sin(np.deg2rad(self.conv_angle))), self.n_points)
            r2 = 2.5*self.rt - np.sqrt((1.5*self.rt)**2 - (l2 - 1.5*self.rt*np.sin(np.deg2rad(self.conv_angle)))**2)
            
            self.vc = self.lstar*self.at
            self.vcyl = self.vc - np.sum(r[0:-1]*r[0:-1])*np.pi*(self.R2*np.sin(np.deg2rad(self.conv_angle)))/(self.n_points-1) - np.sum(r2[0:-1]*r2[0:-1])*np.pi*(1.5*self.rt*np.sin(np.deg2rad(self.conv_angle)))/(self.n_points-1)

            if self.vcyl < 0:
                raise ValueError('L* too short / Contraction ratio too high') 

            self.lcyl = self.vcyl/self.ac
            self.lc = self.lcyl + (self.R2+1.5*self.rt)*np.sin(np.deg2rad(self.conv_angle))
            self.le = (np.sqrt(self.eps)-1)*self.rt*self.rao_percentage/(100*np.tan(np.deg2rad(15)))

            # self.nozzle = Nozzle(
            #     Rt = self.rt * 1000 / 25.4,
            #     CR = self.cr,
            #     eps = self.eps,
            #     pcentBell = self.rao_percentage,
            #     Rup = 1.5,
            #     Rd = 0.382,
            #     Rc = self.R2/self.rt,
            #     cham_conv_ang = self.conv_angle,
            #     theta = None,
            #     exitAng = None,
            #     forceCone = 0,
            #     use_huzel_angles = True)

            self.theta_n = 22 # self.nozzle.theta
            self.theta_e = 14 # self.nozzle.exitAng

            self.generate_contour()

            self.save()
        else:
            self.pe = self.pc/self.cea.get_PcOvPe(Pc=self.pc, MR=self.OF, eps=self.eps)

        # self.print_data()

    def size_injector(self, injector, fuel_Cd, fuel_stiffness, fuel_rho, ox_Cd, ox_stiffness, ox_rho):
        self.fuel_Cd = fuel_Cd
        self.ox_Cd = ox_Cd
        self.fuel_stiffness = fuel_stiffness
        self.fuel_dp = self.fuel_stiffness * self.pc
        self.ox_stiffness = ox_stiffness
        self.ox_dp = self.ox_stiffness * self.pc

        injector.fuel_CdA = self.fuel_mdot / np.sqrt(2*fuel_rho*self.fuel_dp*1e5)
        injector.fuel_A = injector.fuel_CdA / self.fuel_Cd
        injector.ox_CdA = self.ox_mdot / np.sqrt(2*ox_rho*self.ox_dp*1e5)
        injector.ox_A = injector.ox_CdA / self.ox_Cd


        self.fuel_inj_p = self.pc + self.fuel_dp
        self.ox_inj_p = self.pc + self.ox_dp

    def system_combustion_sim(self, fuel, ox, fuel_total_CdA, ox_CdA, fuel_upstream_p, ox_upstream_p, pamb = 1.01325, film_frac = 0, fuel_rho=786, ox_rho=860, ox_gas_class=None, ox_temp=15, fuel_gas_class=None, fuel_temp=15, cstar_eff=1, n_max=100):
        __doc__ = """Combustion sim based on injector pressures.\n
            Required Inputs: fuel, ox, fuel_upstream_p, ox_upstream_p, film_frac, fuel_rho, ox_rho\n
            Optional Inputs: oxclass, ox_gas, ox_temp, fuelclass, fuel_gas, fuel_temp"""

        self.fuel = fuel
        self.ox = ox

        if ox_gas_class is None:
            ox_gas = False
        else:
            ox_gas = True

        if fuel_gas_class is None:
            fuel_gas = False
        else:
            fuel_gas = True

        if ox_gas:
            ox_gas_class.update(Input.temperature(ox_temp), Input.pressure(ox_upstream_p*1e5))
            ox_R = 8.31447/ox_gas_class.molar_mass
            ox_gamma = (ox_gas_class.specific_heat)/(ox_gas_class.specific_heat-ox_R)
            ox_rho = ox_gas_class.density
            # choking_ratio = ((gamma + 1)/2)**(gamma/(gamma-1))
            ox_k = (2/(ox_gamma+1))**((ox_gamma+1)/(ox_gamma-1))
        else:
            ox_gamma = 0
            ox_k = 0

        if fuel_gas:
            fuel_gas_class.update(Input.temperature(fuel_temp), Input.pressure(fuel_upstream_p*1e5))
            fuel_R = 8.31447/fuel_gas_class.molar_mass
            fuel_gamma = (fuel_gas_class.specific_heat)/(fuel_gas_class.specific_heat-fuel_R)
            fuel_rho = fuel_gas_class.density
            fuel_k = (2/(fuel_gamma+1))**((fuel_gamma+1)/(fuel_gamma-1))
        else:
            fuel_gamma = 0
            fuel_k = 0
            
        def pcfunc(pc, cstar, fuel_total_CdA, film_frac, fuel_upstream_p, fuel_rho, ox_CdA, ox_upstream_p, ox_rho, ox_gas, ox_gamma, ox_k, fuel_gas, fuel_gamma, fuel_k):
            if ox_gas:
                ox_min_choked_p = 2 * pc / (2-ox_gamma*ox_k)
                if ox_upstream_p >= ox_min_choked_p:
                    mdot_o = ox_CdA * np.sqrt(ox_gamma * ox_rho * ox_upstream_p * 1e5 * ox_k)
                else:
                    mdot_o = ox_CdA * np.sqrt(2 * ox_rho * (ox_upstream_p - pc) * 1e5)
            else:
                mdot_o = ox_CdA*np.sqrt(2*ox_rho*(ox_upstream_p-pc)*1e5)

            if fuel_gas:
                fuel_min_choked_p = 2 * pc / (2-fuel_gamma*fuel_k)
                if fuel_upstream_p >= fuel_min_choked_p:
                    mdot_f = fuel_total_CdA * np.sqrt(fuel_gamma * fuel_rho * fuel_upstream_p * 1e5 * fuel_k)
                else:
                    mdot_f = fuel_total_CdA * np.sqrt(2 * fuel_rho * (fuel_upstream_p - pc) * 1e5)
            else:
                mdot_f = fuel_total_CdA*np.sqrt(2*fuel_rho*(fuel_upstream_p-pc)*1e5)

            # return ((cstar / self.at) * (mdot_f + mdot_o) * 1e-5) - pc
            return ((cstar / self.at) * ((mdot_f/(1+film_frac)) + mdot_o) * 1e-5) - pc
        
        self.gen_cea_obj()

        min_inj_p = min(fuel_upstream_p, ox_upstream_p)

        cstar_init = 1500
        cstar = cstar_init
        rel_diff = 1

        n = 0
        while rel_diff > 5e-4:
            n += 1
            converged = True
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always", category=RuntimeWarning)
                try:
                    # pc = sp.optimize.fsolve(pcfunc, x0=(1), args=(cstar*cstar_eff, fuel_CdA, film_frac, fuel_upstream_p, fuel_rho, injector.ox_CdA, ox_inj_p, ox_rho, ox_gas, ox_gamma, ox_k, fuel_gas, fuel_gamma, fuel_k))[0]
                    pc = sp.optimize.root_scalar(pcfunc, bracket=[0, min_inj_p], args=(cstar, fuel_total_CdA, film_frac, fuel_upstream_p, fuel_rho, ox_CdA, ox_upstream_p, ox_rho, ox_gas, ox_gamma, ox_k, fuel_gas, fuel_gamma, fuel_k), method='brentq').root
                except ValueError:
                    converged = False
                    pass
                except Exception:
                    converged = False
                    pass
                    
            if ox_gas:
                min_choked_p = 2 * pc / (2-ox_gamma*ox_k)
                if ox_upstream_p >= min_choked_p:
                    mdot_o = ox_CdA * np.sqrt(ox_gamma * ox_rho * ox_upstream_p * 1e5 * ox_k)
                else:
                    mdot_o = ox_CdA * np.sqrt(2 * ox_rho * (ox_upstream_p - pc) * 1e5)
            else:
                mdot_o = ox_CdA*np.sqrt(2*ox_rho*(ox_upstream_p-pc)*1e5)

            if fuel_gas:
                min_choked_p = 2 * pc / (2-fuel_gamma*fuel_k)
                if fuel_upstream_p >= min_choked_p:
                    mdot_f = fuel_total_CdA * np.sqrt(fuel_gamma * fuel_rho * fuel_upstream_p * 1e5 * fuel_k)
                else:
                    mdot_f = fuel_total_CdA * np.sqrt(2 * fuel_rho * (fuel_upstream_p - pc) * 1e5)
            else:
                mdot_f = fuel_total_CdA*np.sqrt(2*fuel_rho*(fuel_upstream_p-pc)*1e5)

            mdot_f /= (1 + film_frac)

            OF = mdot_o / mdot_f
            cstar_old = cstar
            cstar = self.cea.get_Cstar(Pc=pc, MR=OF) * cstar_eff
            rel_diff = abs((cstar - cstar_old) / cstar_old)
            
            if n > n_max:
                print(f"{bcolors.WARNING}Warning: Max iterations exceeded")
                converged = False
                break
        
        if not converged:
            print(f"{bcolors.FAIL}Error: Convergence failed: fuel inj p: {self.fuel_inj_p:.2f}, ox inj p: {self.ox_inj_p:.2f}, n: {n}{bcolors.ENDC}")
        # else:
        #     print(f"Converged: fuel inj p: {self.fuel_inj_p:.2f}, ox inj p: {self.ox_inj_p:.2f}, n: {n}")

        self.combustion_sim(fuel, ox, OF, pc, pamb, cstar_eff)

        self.OF = OF
        self.pc = pc

    def generate_contour(self):
        self.n_points = 20

        l_conv = (self.rc - self.rt) / np.tan(np.deg2rad(self.conv_angle))
        self.r = np.array([self.rc, self.rc])
        self.x = np.array([0, (self.lc - l_conv)])

        # Converging arc
        if self.R2 > 0:
            l = np.linspace(0, (self.R2*np.sin(np.deg2rad(self.conv_angle))), int(1.5*self.n_points))
            r = np.sqrt(self.R2**2 - l**2) + self.rc - self.R2
            l = l + (self.lc - (self.R2+1.5*self.rt)*np.sin(np.deg2rad(self.conv_angle)))
            self.r = np.append(self.r, r[1:])
            self.x = np.append(self.x, l[1:])

        if self.R2 < self.R2_max:
            self.r = np.append(self.r, [self.rt])
            self.x = np.append(self.x, [self.lc])

        if self.rao:
            

            # Throat upstream arc
            l2 = np.linspace(0, (1.5*self.rt*np.sin(np.deg2rad(self.conv_angle))), int(self.n_points))
            r2 = 2.5*self.rt - np.sqrt((1.5*self.rt)**2 - (l2 - 1.5*self.rt*np.sin(np.deg2rad(self.conv_angle)))**2)
            l2 = l2 + self.R2*np.sin(np.deg2rad(self.conv_angle)) + (self.lc - (self.R2+1.5*self.rt)*np.sin(np.deg2rad(self.conv_angle)))
            
            # Throat downstream arc
            l3 = np.linspace(0, (0.382*self.rt*np.sin(np.deg2rad(self.theta_n))), int(0.3*self.n_points))
            r3 = 1.382*self.rt - np.sqrt((0.382*self.rt)**2 - l3**2)
            l3 = l3 + self.lc

            # Parabolic nozzle
            t = np.concatenate((np.linspace(0, 0.2, int(1*self.n_points)), np.linspace(0.2, 1, int(1*self.n_points))))

            Nx = l3[-1]
            Ny = r3[-1]
            Ex = self.lc + self.le
            Ey = self.re

            m1 = np.tan(np.deg2rad(self.theta_n))
            m2 = np.tan(np.deg2rad(self.theta_e))
            c1 = Ny - m1*Nx
            c2 = Ey - m2*Ex
            Qx = (c2 - c1)/(m1 - m2)
            Qy = (m1*c2 - m2*c1)/(m1 - m2)

            l4 = Nx*(1-t)**2 + 2*(1-t)*t*Qx + Ex*t**2
            r4 = Ny*(1-t)**2 + 2*(1-t)*t*Qy + Ey*t**2

            self.r = np.concatenate((self.r, r2[1:], r3[1:], r4[1:]))
            self.x = np.concatenate((self.x, l2[1:], l3[1:], l4[1:]))
        
        else:
            # throat + diverging section
            self.r = np.append(self.r, [self.rt, self.re])
            self.x = np.append(self.x, [self.ln + self.lc, self.ltotal])

        self.x = self.x - self.lc
        self.stations = len(self.r)

    def show_contour(self):
        self.generate_contour()
        plt.figure()
        plt.plot(self.x*1e3, self.r*1e3, 'b', label='Contour')
        plt.plot(self.x*1e3, -self.r*1e3, 'b')
        plt.plot(self.x*1e3, -self.r*1e3, 'xk')
        plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
        plt.grid()
        plt.xlabel('Axial Distance (mm)')
        plt.ylabel('Radius (mm)')
        plt.title('Chamber Contour')
        plt.gca().set_aspect('equal', adjustable='datalim')
        plt.xlim(left=self.x[0]*1e3, right=self.x[-1]*1e3)
        plt.legend()

    def save(self):
        with open(self.file, 'w+') as output_file:
            output_file.write(f'dc: {self.dc*1e3}')
            output_file.write(f'\ndt: {self.dt*1e3}')
            output_file.write(f'\nde: {self.de*1e3}')
            output_file.write(f'\neps: {self.eps}')
            output_file.write(f'\nconv_angle: {self.conv_angle}')
            output_file.write(f'\nlc: {self.lc*1e3}')
            output_file.write(f'\nle: {self.le*1e3}')
            output_file.write(f'\ntheta_n: {self.theta_n}')
            output_file.write(f'\ntheta_e: {self.theta_e}')

    def print_data(self):
        print('-----------------------------------')
        print(f'{self.fuel} / {self.ox}\n')
        print(f'{"Parameter":<20} {"Value":<10} {"Unit"}')
        print(f'{"OF:":<20} {self.OF:<10.3f}')
        print(f'{"Chamber Pressure:":<20} {self.pc:<10.2f} bar')
        print(f'{"Ambient Pressure:":<20} {self.pamb:<10.2f} bar')
        print(f'{"Thrust:":<20} {self.thrust:<10.2f} N')
        print(f'{"ISP:":<20} {self.isp:<10.2f} s')
        print(f'{"SL ISP:":<20} {self.ispsea:<10.2f} s')
        print(f'{"Vac ISP:":<20} {self.ispvac:<10.2f} s')
        print(f'{"C*:":<20} {self.cstar:<10.2f} m/s')
        print(f'{"Cf:":<20} {self.cf:<10.4f}\n')
        print(f'{"Chamber Temp:":<20} {self.Tg_c:<10.1f} K')
        print(f'{"Throat Temp:":<20} {self.Tg_t:<10.1f} K')
        print(f'{"Exit Temp:":<20} {self.Tg_e:<10.1f} K\n')
        print(f'{"Throat Pressure:":<20} {self.pt:<10.2f} bar')
        print(f'{"Exit Pressure:":<20} {self.pe:<10.2f} bar')
        print(f'{"Pc loss ratio:":<20} {self.PinjPcomb:<10.3f}\n')
        print(f'{"Exit Mach Number:":<20} {self.Me:<10.3f}')
        print(f'{"Expansion Ratio:":<20} {self.eps:<10.3f}')
        print(f'{"Exit Condition:":<20} {self.exitcond:<10}\n')
        if(self.mdot >= 1e-1):
            print(f'{"Total mdot:":<20} {self.mdot:<10.4f} kg/s')
            print(f'{"Ox mdot:":<20} {self.ox_mdot:<10.4f} kg/s')
            print(f'{"Fuel mdot:":<20} {self.fuel_mdot:<10.4f} kg/s\n')
        else:
            print(f'{"Total mdot:":<20} {self.mdot*1e3:<10.4f} g/s')
            print(f'{"Ox mdot:":<20} {self.ox_mdot*1e3:<10.4f} g/s')
            print(f'{"Fuel mdot:":<20} {self.fuel_mdot*1e3:<10.4f} g/s\n')
        print(f'{"Chamber Diameter:":<20} {self.dc*1e3:<10.3f} mm')
        print(f'{"Throat Diameter:":<20} {self.dt*1e3:<10.3f} mm')
        print(f'{"Exit Diameter:":<20} {self.de*1e3:<10.3f} mm\n')
        print(f'{"Chamber Length:":<20} {self.lc*1e3:<10.3f} mm')
        if self.rao == False:
            print(f'{"Nozzle Length:":<20} {self.ln*1e3:<10.3f} mm')
        print(f'{"Exit Length:":<20} {self.le*1e3:<10.3f} mm')
        print(f'{"Total Length:":<20} {self.ltotal*1e3:<10.3f} mm\n')
        print(f'{"Contraction Ratio:":<20} {self.cr:<10.3f}')
        print(f'{"Converging Angle:":<20} {self.conv_angle:<10.3f}°')
        if self.rao:
            print(f'{"Nozzle Inlet Angle:":<20} {self.theta_n:<10.3f}°')
            print(f'{"Nozzle Exit Angle:":<20} {self.theta_e:<10.3f}°')
        else:
            print(f'{"Diverging Angle:":<20} {self.div_angle:<10.3f}°')
        print('-----------------------------------')

    def thermal_sim(self, wall_k, n_channels, h_rib, tw, channel_arc_angle, coolant, coolant_mdot, coolant_T_in, coolant_p_in, rev_flow):
        # Seems way off at higher pc with way too high wall temp
        self.generate_contour()

        self.wall_k = wall_k
        self.coolant = coolant
        self.n_channels = n_channels
        self.h_rib = h_rib
        self.tw = tw
        self.channel_arc_angle = channel_arc_angle
        self.channel_width = (2 * self.r + 2 * self.tw + self.h_rib) * np.sin(np.deg2rad(channel_arc_angle / 2))
        self.channel_area = self.channel_width * self.h_rib
        self.d_h = np.sqrt(4 * self.channel_area / np.pi)
        self.coolant_mdot = coolant_mdot

        self.pr          = np.zeros(self.stations)
        self.gamma       = np.zeros(self.stations)
        self.M           = np.zeros(self.stations)
        self.hg          = np.zeros(self.stations)
        self.q           = np.zeros(self.stations)
        self.pg          = np.zeros(self.stations)
        self.Tg          = np.zeros(self.stations)
        self.Twg         = np.zeros(self.stations)
        self.Twc         = np.zeros(self.stations)
        self.Tc          = np.zeros(self.stations)
        self.v_coolant   = np.zeros(self.stations)
        self.rho_coolant = np.zeros(self.stations)
        self.mu_coolant  = np.zeros(self.stations)
        self.k_coolant   = np.zeros(self.stations)
        self.cp_coolant  = np.zeros(self.stations)
        self.Re_coolant  = np.zeros(self.stations)
        self.Pr_coolant  = np.zeros(self.stations)
        self.Nu_coolant  = np.zeros(self.stations)
        self.hc          = np.zeros(self.stations)
        self.total_heat_flux = 0

        def machfunc(mach, area, gamma):
            area_ratio = area / self.at
            if mach == 0:
                mach = 1e-7
            return (area_ratio - ((1.0/mach) * ((1 + 0.5*(gamma-1)*mach*mach) / ((gamma + 1)/2))**((gamma+1) / (2*(gamma-1)))))

        t_next = coolant_T_in

        self.coolant_class = Fluid(self.coolant)
        self.coolant_class.update(Input.pressure(coolant_p_in*1e5),  Input.temperature(coolant_T_in-273.15))
        coolant_heat = 0

        for j in range(self.stations):
            if rev_flow == True:
                i = -(j+1)
                # i = -j
                inext = i - 1
            else:
                i = j
                inext = i + 1
            r = self.r[i]
            A = np.pi * r * r
            self.Tc[i] = t_next

            self.coolant_class.update(Input.pressure(coolant_p_in*1e5),  Input.temperature(self.Tc[i] - 273.15))
            # self.coolant_class.update(Input.enthalpy(self.coolant_class.enthalpy + coolant_heat),  Input.pressure(coolant_p_in*1e5)) 
            
            self.rho_coolant[i] = self.coolant_class.density 
            self.cp_coolant[i] = self.coolant_class.specific_heat
            self.mu_coolant[i] = self.coolant_class.dynamic_viscosity
            self.k_coolant[i] = self.coolant_class.conductivity
            self.v_coolant[i] = self.coolant_mdot / (self.rho_coolant[i] * self.channel_area[i] * self.n_channels)
            self.Re_coolant[i] = self.rho_coolant[i] * self.v_coolant[i] * self.d_h[i] / self.mu_coolant[i]
            self.Pr_coolant[i] = self.mu_coolant[i] * self.cp_coolant[i] / self.k_coolant[i]
            self.Nu_coolant[i] = 0.023 * self.Re_coolant[i]**0.8 * self.Pr_coolant[i]**0.4 # Dittus-Boelter
            self.hc[i] = self.Nu_coolant[i] * self.k_coolant[i] / self.d_h[i]

            # if self.x[i] == 0: # Throat
            #     self.gamma[i] = self.gam_t
            #     self.pr[i] = self.pr_t
            #     self.M[i] = 1
            throat_rcurv =  self.rt * (0.382 + 1.5)/2
            if self.x[i] < 0: # Converging section
                self.gamma[i] = (self.gam_t - self.gam_c)/(self.rt - self.rc) * (r - self.rc) + self.gam_c
                self.pr[i] = (self.pr_t - self.pr_c)/(self.rt - self.rc) * (r - self.rc) + self.pr_c
                self.M[i] = root_scalar(machfunc, args=(A, self.gamma[i]), bracket=[0, 1]).root
                # throat_rcurv =  self.rt * 0.382
            else: # Diverging section
                self.gamma[i] = (0.5*(0.8*self.gam_e+1.2*self.gam_t) - self.gam_t)/(self.re - self.rt) * (r - self.rt) + self.gam_t
                self.pr[i] = (self.pr_e - self.pr_t)/(self.re - self.rt) * (r - self.rt) + self.pr_t
                self.M[i] = root_scalar(machfunc, args=(A, self.gamma[i]), bracket=[1, 5]).root
                # throat_rcurv =  self.rt * 1.5

            self.Tg[i] = self.Tg_c * ((1 + (0.5 * (self.pr[i]**(1/3)) * (self.gamma[i] - 1) * (self.M[i]**2))) / (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2))
            # self.Tg[i] = self.Tg_c / (1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2)
            self.pg[i] = self.pc * (self.Tg[i] / self.Tg_c)**(self.gamma[i] / (self.gamma[i] - 1))

            bartz = ((0.026 / (self.dt**0.2))) * ((self.mu_c**0.2) * self.cp_c / (self.pr_c**0.6)) * ((self.pc * self.cstar)**0.8) * ((self.dt / throat_rcurv)**0.1) * ((self.at / A)**0.9)

            if j != self.stations - 1:
                dA = np.pi * (self.r[i] + self.r[inext]) * np.sqrt((self.r[i] - self.r[inext]) ** 2 + (self.x[inext] - self.x[i]) ** 2)

            def wall_temp_func(Twg):
                correction_factor = (((0.5 * (Twg / self.Tg_c) * (1 + (0.5 * (self.gamma[i] - 1)) * (self.M[i]**2)) + 0.5)**0.68) * ((1 + 0.5 * (self.gamma[i] - 1) * (self.M[i]**2))**0.12))**-1
                hg = bartz * correction_factor * 0.369375
                q = hg * (self.Tg[i] - Twg)
                Twc = (q / self.hc[i]) + self.Tc[i]
                Twg_new = (q * self.tw / self.wall_k) + Twc
                return Twg_new - Twg

            bracket = [self.Tc[i], self.Tg[i]]
            sol = root_scalar(wall_temp_func, bracket=bracket, method='brentq')
            if not sol.converged:
                print(f"Warning: Wall temp solver did not converge at station {i}")
                self.Twg[i] = 0
            else:
                self.Twg[i] = sol.root

            correction_factor = (((0.5 * (self.Twg[i] / self.Tg_c) * (1 + (0.5 * (self.gamma[i] - 1) * self.M[i]**2)) + 0.5)**0.68) * ((1 + 0.5 * (self.gamma[i] - 1) * self.M[i]**2))**0.12)**-1
            self.hg[i] = bartz * correction_factor * 0.369375
            self.q[i] = self.hg[i] * (self.Tg[i] - self.Twg[i])
            self.Twc[i] = (self.q[i] / self.hc[i]) + self.Tc[i]
            self.Twg[i] = (self.q[i] * self.tw / self.wall_k) + self.Twc[i]

            coolant_heat = self.q[i] * dA / self.coolant_mdot

            t_next = (coolant_heat / self.cp_coolant[i]) + self.Tc[i]

            self.total_heat_flux += self.q[i] * dA

        # if rev_flow == True:
        #     self.hg[0] = self.hg[1]
        #     self.q[0] = self.q[1]
        #     self.Twg[0] = self.Twg[1]
        #     self.Twc[0] = self.Twc[1]
        #     self.Tc[0] = t_next
        # else:
        #     print(f'hg[-2]: {self.hg[-2]}')
        #     print(f'hg[-1]: {self.hg[-1]}')
        #     self.hg[-1] = self.hg[-2]
        #     self.q[-1] = self.q[-2]
        #     self.Twg[-1] = self.Twg[-2]
        #     self.Twc[-1] = self.Twc[-2]
        #     self.Tc[-1] = t_next
            
    def plot_thermals(self, title):
        self.thermalsplot = subplot(3, 5, title, self)
        xlabel = 'Axial Distance (mm)'
        self.thermalsplot.plt(1, self.x*1e3, self.hg*1e-3,'Gas Side Conv. Coeff.', xlabel, 'Gas Side Conv. Coeff. (kW/m^2/K)','r', True)
        self.thermalsplot.plt(2, self.x*1e3, self.q*1e-3,'Heat Flux', xlabel, 'Heat Flux (kW/m^2)','r', True)
        self.thermalsplot.plt(3, self.x*1e3, self.Tg, 'Gas Temperature', xlabel, 'Gas Temperature (K)', 'r', True)
        self.thermalsplot.plt(4, self.x*1e3, self.pg, 'Pressure', xlabel, 'Pressure (bar)', 'b', True)
        self.thermalsplot.plt(5, self.x*1e3, self.M, 'Mach', xlabel, 'Mach', 'b', True)
        self.thermalsplot.plt(6, self.x*1e3, self.gamma, 'Gamma', xlabel, 'Gamma', 'b', True)
        self.thermalsplot.plt(7, self.x*1e3, self.pr, 'Prandtl Number', xlabel, 'Prandtl Number', 'b', True)
        self.thermalsplot.plt(8, self.x*1e3, self.Twg, 'Wall Temp', xlabel, 'Wall Temp (K)', 'r', True, label='Twg')
        self.thermalsplot.addline(8, self.x*1e3, self.Twc, 'm', label='Twc')
        self.thermalsplot.addline(8, self.x*1e3, self.Tc, 'b', label='Tc')
        self.thermalsplot.plt(9, self.x*1e3, self.Tc, 'Coolant Temperature', xlabel, 'Coolant Temperature (K)', 'r', True)
        self.thermalsplot.plt(10, self.x*1e3, self.rho_coolant, 'Coolant Density', xlabel, 'Coolant Density (kg/m^3)', 'b', True)
        self.thermalsplot.plt(11, self.x*1e3, self.hc*1e-3,'Coolant Side Conv. Coeff.', xlabel, 'Coolant Side Conv. Coeff. (kW/m^2/K)', 'r', True)
        self.thermalsplot.plt(12, self.x*1e3, self.v_coolant, 'Coolant Velocity', xlabel, 'Coolant Velocity (m/s)', 'b', True)
        self.thermalsplot.plt(13, self.x*1e3, self.cp_coolant, 'Coolant Specific Heat', xlabel, 'Coolant Specific Heat (J/kg/K)', 'b', True)
        self.thermalsplot.plt(14, self.x*1e3, self.k_coolant, 'Coolant Thermal Conductivity', xlabel, 'Coolant Thermal Conductivity (W/m/K)', 'b', True)
        self.thermalsplot.plt(15, self.x*1e3, self.mu_coolant*1e3, 'Coolant Viscosity', xlabel, 'Coolant Viscosity (mPa.s)', 'b', True)

    def plot_film(self):
        self.filmplot = subplot(2,2,'Film Cooling Data', self)
        xlabel = 'Axial Distance (mm)'
        self.filmplot.plt(1, self.x*1e3, self.T_f, 'Film Temp', xlabel, 'Film Temperature (K)', 'r', True)
        self.filmplot.plt(2, self.x*1e3, self.q_f*1e-3, 'Film Heat Flux', xlabel, 'Film Heat Flux (kW/m^2)', 'r', True)
        self.filmplot.plt(3, self.x*1e3, self.film_mdot_l, 'Liquid Film mdot', xlabel, 'Liquid Film mdot (kg/s)', 'b', True,label='Liquid Film mdot (kg/s)')
        self.filmplot.addline(3, self.x*1e3, self.film_mdot_g, 'r', label='Gas Film mdot (kg/s)')
        self.filmplot.plt(4, self.x*1e3, self.filmstate, 'Film State', xlabel, 'Film State', 'b', True)#

    def system_sim_sensitivity(self, param, param_range, fuel, ox, fuel_inj_p, ox_inj_p, fuel_rho, ox_rho, oxclass, ox_can_choke=False, ox_temp=15, fuel_can_choke=False, fuel_temp=15):
        pc = np.array([])
        thrust = np.array([])
        OF = np.array([])
        ox_mdot = np.array([])
        fuel_mdot = np.array([])
        ox_inj_p_arr = np.array([])
        fuel_inj_p_arr = np.array([])

        try:
            self.system_combustion_sim(
                fuel = fuel,
                ox = ox,
                fuel_upstream_p = fuel_inj_p,
                ox_upstream_p = ox_inj_p,
                fuel_rho = fuel_rho,
                ox_rho = ox_rho,
                ox_gas_class = oxclass,
                ox_gas = ox_can_choke,
                ox_temp = ox_temp,
                fuel_gas = fuel_can_choke,
            )
            pc = np.append(pc, self.pc)
            thrust = np.append(thrust, self.thrust)
            OF = np.append(OF, self.OF)
            ox_mdot = np.append(ox_mdot, self.ox_mdot)
            fuel_mdot = np.append(fuel_mdot, self.fuel_mdot)
        except ValueError:
            pc = np.append(pc, 0)
            thrust = np.append(thrust, 0)
            OF = np.append(OF, 0)
            ox_mdot = np.append(ox_mdot, 0)
            fuel_mdot = np.append(fuel_mdot, 0)
        ox_inj_p_arr = np.append(ox_inj_p_arr, self.ox_inj_p)
        fuel_inj_p_arr = np.append(fuel_inj_p_arr, self.fuel_inj_p)

        total_mdot = ox_mdot + fuel_mdot
        sens_study = plt.figure(constrained_layout=True)
        sens_study.suptitle(f'{param} Sensitivity Study')
        ax1 = sens_study.add_subplot(2, 2, 1)
        ax1.plot(param_range, pc, 'k')
        ax1.plot(param_range, (fuel_inj_p_arr-pc), 'r')
        # ax1.plot(param_range, (ox_inj_p_arr-pc), 'b')
        ax1.set_title('Pressures')
        ax1.set_xlabel(param)
        ax1.set_ylabel('Chamber Pressure (bar)')
        ax1.legend(['Chamber', 'Fuel dP', 'Ox dP'])
        ax1.grid(alpha=1)
        ax2 = sens_study.add_subplot(2, 2, 2)
        ax2.plot(param_range, thrust, 'r')
        ax2.set_title('Thrust')
        ax2.set_xlabel(param)
        ax2.set_ylabel('Thrust (N)')
        ax2.grid(alpha=1)
        ax3 = sens_study.add_subplot(2, 2, 3)
        ax3.plot(param_range, OF, 'r')
        ax3.set_title('O/F Ratio')
        ax3.set_xlabel(param)
        ax3.set_ylabel('O/F Ratio')
        ax3.grid(alpha=1)
        ax4 = sens_study.add_subplot(2, 2, 4)
        ax4.plot(param_range, total_mdot*1e3, 'k')
        ax4.plot(param_range, ox_mdot*1e3, 'b')
        ax4.plot(param_range, fuel_mdot*1e3, 'r')
        ax4.set_title('Mass Flow Rate')
        ax4.set_xlabel(param)
        ax4.set_ylabel('Mass Flow Rate (g/s)')
        ax4.grid(alpha=1)
        ax4.legend(['Total', 'Ox', 'Fuel'])

class injector():
    __doc__ = """A class representing a bipropellant injector for a rocket engine.
        The injector class provides methods to size both fuel and oxidizer injector elements 
        and calculate propellant flow rates through these elements. It supports various injector 
        configurations including annular holes and circular holes.

        Attributes
        fuel_A : float
            Fuel injector total area in square meters.

        fuel_Cd : float
            Fuel injector discharge coefficient, default 0.75.

        ox_A : float
            Oxidizer injector total area in square meters.

        ox_Cd : float
            Oxidizer injector discharge coefficient, default 0.4.

        fuel_CdA : float
            Product of fuel discharge coefficient and area.

        ox_CdA : float
            Product of oxidizer discharge coefficient and area.

        Methods

        size_fuel_anulus(Cd, ID, OD, n=1)
            Sizes fuel injector for annular holes.

        size_ox_anulus(Cd, ID, OD, n=1)
            Sizes oxidizer injector for annular holes.

        size_fuel_holes(Cd, d, n=1)
            Sizes fuel injector for circular holes.

        size_ox_holes(Cd, d, n=1)
            Sizes oxidizer injector for circular holes.

        spi_fuel_mdot(dp, fuel_rho)
            Calculates fuel mass flow rate using single phase incompressible model.

        spi_ox_mdot(dp, ox_rho)
            Calculates oxidizer mass flow rate using single phase incompressible model.

        calc_start_mdot(fuel_inj_p, ox_inj_p, fuel_rho=786, ox_rho=860, ox_gas_class=None, 
                    ox_temp=15, fuel_gas_class=None, fuel_temp=15)
            Calculates starting mass flow rates for injector venting to atmosphere.
            Supports both incompressible liquid and compressible gas calculations.
        Notes
        -----
        The class converts input dimensions in millimeters to meters internally.
        For gas propellants, the class can calculate choked flow conditions."""

    def __init__(self):
        self.film_CdA = 0
        self.film_frac = 0

    def calc_film(self):
        __doc__ = """
            Calculates the film cooling fraction and total fuel CdA.
            """
        if self.fuel_core_CdA is None:
            raise ValueError("Core fuel CdA must be set before calculating film fraction.")
        self.film_frac = self.film_CdA / self.fuel_core_CdA
        self.fuel_total_CdA = self.fuel_core_CdA + self.film_CdA

    def set_fuel_CdA(self, CdA):
        __doc__ = """
            Sets the fuel injector CdA.

            Parameters
            ----------
            CdA : float
                Product of fuel discharge coefficient and area.
            """
        self.fuel_core_CdA = CdA
        self.calc_film()

    def set_ox_CdA(self, CdA):
        __doc__ = """
            Sets the oxidizer injector CdA.

            Parameters
            ----------
            CdA : float
                Product of oxidizer discharge coefficient and area.
            """
        self.ox_CdA = CdA

    def set_film_CdA(self, CdA):
        __doc__ = """
            Sets the film cooling injector CdA.

            Parameters
            ----------
            CdA : float
                Product of film cooling discharge coefficient and area.
            """
        self.film_CdA = CdA
        self.calc_film()

    def size_fuel_anulus(self, Cd, ID, OD, n = 1):
        __doc__ = """
            Sizes fuel injector for a number of identical annular holes.

            Parameters
            ----------
            Cd : float
                Discharge coefficient for the fuel annulus.
            ID : float
                Inner diameter of the annulus in millimeters.
            OD : float
                Outer diameter of the annulus in millimeters.
            n : int, optional
                Number of annular holes (default 1).
            """
        self.fuel_Cd = Cd
        self.fuel_A = 0.25e-6 * np.pi * (OD**2 - ID**2) * n
        self.fuel_core_CdA = self.fuel_A * Cd
        self.calc_film()
    
    def size_ox_anulus(self, Cd, ID, OD, n = 1):
        __doc__ = """
            Sizes oxidizer injector for a number of identical annular holes.

            Parameters
            ----------
            Cd : float
                Discharge coefficient for the oxidizer annulus.
            ID : float
                Inner diameter of the annulus in millimeters.
            OD : float
                Outer diameter of the annulus in millimeters.
            n : int, optional
                Number of annular holes (default 1).
            """
        self.ox_Cd = Cd
        self.ox_A = 0.25e-6 * np.pi * (OD**2 - ID**2) * n
        self.ox_CdA = self.ox_A * Cd

    def size_fuel_holes(self, Cd, d, n = 1):
        __doc__ = """
            Sizes the fuel injector for a number of identical holes.

            Parameters
            ----------
            Cd : float
                Discharge coefficient for the fuel holes.
            d : float
                Hole diameter in millimeters.
            n : int, optional
                Number of fuel holes (default 1).
            """
        self.fuel_Cd = Cd
        self.fuel_A = 0.25e-6 * np.pi * (d**2) * n
        self.fuel_core_CdA = self.fuel_A * Cd
        self.calc_film()
    
    def size_film_holes(self, Cd, d, n = 1):
        __doc__ = """
            Sizes the film cooling injector for a number of identical holes.

            Parameters
            ----------
            Cd : float
                Discharge coefficient for the film cooling holes.
            d : float
                Hole diameter in millimeters.
            n : int, optional
                Number of film cooling holes (default 1).
            """
        self.film_Cd = Cd
        self.film_A = 0.25e-6 * np.pi * (d**2) * n
        self.film_CdA = self.film_A * Cd
        self.calc_film()

    def size_ox_holes(self, Cd, d, n = 1):
        __doc__ = """
            Sizes the oxidizer injector for a number of identical holes.

            Parameters
            ----------
            Cd : float
                Discharge coefficient for the oxidizer holes.
            d : float
                Hole diameter in millimeters.
            n : int, optional
                Number of oxidizer holes (default 1).
            """
        self.ox_Cd = Cd
        self.ox_A = 0.25e-6 * np.pi * (d**2) * n
        self.ox_CdA = self.ox_A * Cd

    def spi_fuel_core_mdot(self, dp, fuel_rho):
        __doc__ = """
            Calculates the core fuel mass flow rate through the injector using the single phase incompressible model.

            Parameters
            ----------
            dp : float
                Pressure differential across the injector orifice (bar)
            fuel_rho : float
                Density of the fuel (kg/m^3)

            Returns
            -------
            float
                Fuel mass flow rate (kg/s)
            """
        return self.fuel_core_CdA * np.sqrt(2e5 * dp * fuel_rho)
    
    def spi_fuel_total_mdot(self, dp, fuel_rho):
        __doc__ = """
            Calculates the total fuel mass flow rate through the injector using the single phase incompressible model.

            Parameters
            ----------
            dp : float
                Pressure differential across the injector orifice (bar)
            fuel_rho : float
                Density of the fuel (kg/m^3)

            Returns
            -------
            float
                Total fuel mass flow rate (kg/s)
            """
        return self.fuel_total_CdA * np.sqrt(2e5 * dp * fuel_rho)

    def spi_film_mdot(self, dp, film_rho):
        __doc__ = """
            Calculates the film cooling mass flow rate through the injector using the single phase incompressible model.

            Parameters
            ----------
            dp : float
                Pressure differential across the injector orifice (bar)
            film_rho : float
                Density of the film coolant (kg/m^3)

            Returns
            -------
            float
                Film cooling mass flow rate (kg/s)
            """
        return self.film_CdA * np.sqrt(2e5 * dp * film_rho)

    def spi_ox_mdot(self, dp, ox_rho):
        __doc__ = """
            Calculates the oxidiser mass flow rate through the injector using the single phase incompressible model.

            Parameters
            ----------
            dp : float
                Pressure differential across the injector orifice (bar)
            ox_rho : float
                Density of the oxidiser (kg/m^3)

            Returns
            -------
            float
                Oxidiser mass flow rate (kg/s)
            """
        return self.ox_CdA * np.sqrt(2e5 * dp * ox_rho)

    def spi_fuel_core_dp(self, mdot, fuel_rho):
        __doc__ = """
            Calculates the pressure differential across the fuel injector orifice using the single phase incompressible model.

            Parameters
            ----------
            mdot : float
                Fuel mass flow rate (kg/s)
            fuel_rho : float
                Density of the fuel (kg/m^3)

            Returns
            -------
            float
                Pressure differential across the injector orifice (bar)
            """
        return ((mdot / self.fuel_core_CdA)**2) / (2e5 * fuel_rho)

    def spi_fuel_total_dp(self, mdot, fuel_rho):
        __doc__ = """
            Calculates the pressure differential across the total fuel injector orifice using the single phase incompressible model.

            Parameters
            ----------
            mdot : float
                Total fuel mass flow rate (kg/s)
            fuel_rho : float
                Density of the fuel (kg/m^3)

            Returns
            -------
            float
                Pressure differential across the injector orifice (bar)
            """
        return ((mdot / self.fuel_total_CdA)**2) / (2e5 * fuel_rho)

    def spi_film_dp(self, mdot, film_rho):
        __doc__ = """
            Calculates the pressure differential across the film cooling injector orifice using the single phase incompressible model.

            Parameters
            ----------
            mdot : float
                Film cooling mass flow rate (kg/s)
            film_rho : float
                Density of the film coolant (kg/m^3)

            Returns
            -------
            float
                Pressure differential across the injector orifice (bar)
            """
        return ((mdot / self.film_CdA)**2) / (2e5 * film_rho)
   
    def spi_ox_dp(self, mdot, ox_rho):
        __doc__ = """
            Calculates the pressure differential across the oxidizer injector orifice using the single phase incompressible model.

            Parameters
            ----------
            mdot : float
                Oxidizer mass flow rate (kg/s)
            ox_rho : float
                Density of the oxidizer (kg/m^3)

            Returns
            -------
            float
                Pressure differential across the injector orifice (bar)
            """
        return ((mdot / self.ox_CdA)**2) / (2e5 * ox_rho)
    
    def ox_flow_setup(self, ox_class, ox_downstream_p, ox_upstream_p, ox_upstream_T, ox_vp):
        self.ox_downstream_p = ox_downstream_p
        self.ox_upstream_p = ox_upstream_p
        self.ox_upstream_T = ox_upstream_T
        self.ox_vp = ox_vp

        self.ox_downstream_p *= 1e5

        self.ox_up = Fluid(ox_class)
        self.ox_down = Fluid(ox_class)

        if ox_upstream_p == None:
            self.ox_saturated = True
        else:
            self.ox_saturated = False
            self.ox_upstream_p *= 1e5
        
        if self.ox_upstream_T == None and self.ox_vp == None:
            raise ValueError("Either upstream_T or vp must be provided.")
        elif self.ox_upstream_T is not None and self.ox_vp is not None:
            raise ValueError("Both upstream_T and vp cannot be provided.")
        elif self.ox_upstream_T is not None:
            if self.ox_saturated:
                self.ox_up.update(Input.temperature(self.ox_upstream_T), Input.quality(0))
                self.ox_vp = self.ox_up.pressure
                self.ox_upstream_p = self.ox_vp
            else:
                self.ox_up.update(Input.temperature(self.ox_upstream_T), Input.pressure(self.ox_upstream_p))
        elif self.ox_vp is not None:
            self.ox_vp *= 1e5
            self.ox_up.update(Input.pressure(self.ox_vp), Input.quality(0))
            self.ox_upstream_T = self.ox_up.temperature
            if self.ox_saturated:
                self.ox_upstream_p = self.ox_vp + 10
            else:
                self.ox_up.update(Input.temperature(self.ox_upstream_T), Input.pressure(self.ox_upstream_p))
        
        self.ox_down.update(Input.pressure(self.ox_downstream_p), Input.entropy(self.ox_up.entropy))

    def hem_ox_mdot(self, ox_class, downstream_p, upstream_p=None, upstream_T = None, vp = None):
        __doc__ = """
            Calculates the oxidizer mass flow rate using the HEM model.
            Used for fluids that can exhibit two phase flow.

            Parameters
            ----------
            ox_class : object
                pyfluids object for oxidizer
            downstream_p : float
                Downstream pressure (bar)
            upstream_p : float, optional
                Upstream pressure (bar), defaults to None
            upstream_T : float, optional
                Upstream temperature (°C), defaults to None
            vp : float, optional
                Vapor pressure (bar), defaults to None

            Returns
            -------
            float
                Oxidizer mass flow rate (kg/s)
            """

        self.ox_flow_setup(ox_class, downstream_p, upstream_p, upstream_T, vp)

        def HEMfunc(up, down, downstream_p):
            down.update(Input.pressure(downstream_p), Input.entropy(up.entropy))
            return self.ox_CdA * down.density * np.sqrt(2 * (up.enthalpy - down.enthalpy))

        sol = sp.optimize.minimize_scalar(lambda x: -HEMfunc(self.ox_up, self.ox_down, x), bounds=[0,self.ox_upstream_p], method='bounded')

        self.ox_choked_p = sol.x
        choked_mdot = -sol.fun

        if (self.ox_choked_p > self.ox_downstream_p):
            mdot = choked_mdot
        else:
            mdot = HEMfunc(self.ox_up, self.ox_down, self.ox_downstream_p)

        mdot = 0 if np.isnan(mdot) else mdot

        return mdot

    def nhne_ox_mdot(self, ox_class, downstream_p, upstream_p=None, upstream_T = None, vp = None):
        __doc__ = """
            Calculates the oxidizer mass flow rate using the NHNE model.
            Used for fluids that can exhibit two phase flow.

            Parameters
            ----------
            ox_class : object
                pyfluids object for oxidizer
            downstream_p : float
                Downstream pressure (bar)
            upstream_p : float, optional
                Upstream pressure (bar), defaults to None
            upstream_T : float, optional
                Upstream temperature (°C), defaults to None
            vp : float, optional
                Vapor pressure (bar), defaults to None

            Returns
            -------
            float
                Oxidizer mass flow rate (kg/s)
            """

        HEM_mdot = self.hem_ox_mdot(ox_class, downstream_p, upstream_p, upstream_T, vp)
        SPI_mdot = self.spi_ox_mdot((self.ox_upstream_p - self.ox_downstream_p)/1e5, self.ox_up.density)

        k = np.sqrt((self.ox_upstream_p - self.ox_downstream_p) / (self.ox_vp - self.ox_downstream_p)) if self.ox_downstream_p < self.ox_vp else 1

        mdot = (SPI_mdot* k / (1 + k)) + (HEM_mdot / (1 + k))

        return mdot

    def calc_start_mdot(self, fuel_inj_p, ox_inj_p, fuel_rho=786, ox_rho=860, ox_gas_class=None, ox_temp=15, fuel_gas_class=None, fuel_temp=15):
        __doc__ = """
            Calculates the starting mdots for the injector (venting to atm).
            Disregards film cooling.
            ----------
            fuel_inj_p : float
                Fuel injector pressure (bar)
            ox_inj_p : float
                Oxidizer injector pressure (bar)
            fuel_rho : float, optional
                Fuel density in kg/m³, defaults to 786
            ox_rho : float, optional
                Oxidizer density in kg/m³, defaults to 860 (used only if oxclass is None)
            oxclass : object, optional
                pyfluids object for oxidizer
                If provided, compressible flow calculations will be used for oxidizer
            ox_temp : float, optional
                Oxidizer temperature in °C, defaults to 15 (used only if oxclass is provided)
            Returns
            -------
            None
                Results are printed directly:
                - Total mass flow rate (g/s)
                - Oxidizer mass flow rate (g/s) and whether flow is choked
                - Fuel mass flow rate (g/s)
                - Oxidizer to fuel ratio (OF)
            """
        ox_chokedstate = 'Unchoked'
        fuel_chokedstate = 'Unchoked'
        if ox_gas_class != None:
            ox_gas_class.update(Input.temperature(ox_temp), Input.pressure(ox_inj_p*1e5))
            ox_R = 8.31447/ox_gas_class.molar_mass
            ox_gamma = (ox_gas_class.specific_heat)/(ox_gas_class.specific_heat-ox_R)
            ox_rho = ox_gas_class.density
            # choking_ratio = ((ox_gamma + 1)/2)**(ox_gamma/(ox_gamma-1))
            ox_k = (2/(ox_gamma+1))**((ox_gamma+1)/(ox_gamma-1))
            min_choked_p = 2 * 1.01325 / (2-ox_gamma*ox_k)
            if ox_inj_p >= min_choked_p:
                ox_mdot_start = self.ox_CdA * np.sqrt(ox_gamma*ox_rho*ox_inj_p*1e5*ox_k)
                ox_chokedstate = 'Choked'
            else:
                ox_mdot_start = self.ox_CdA * np.sqrt(2*(ox_inj_p-1.01325) * 1e5 * ox_rho)
        else:
            ox_mdot_start = self.ox_CdA * np.sqrt(2*(ox_inj_p-1.01325) * 1e5 * ox_rho)

        if fuel_gas_class != None:
            fuel_gas_class.update(Input.temperature(fuel_temp), Input.pressure(fuel_inj_p*1e5))
            fuel_R = 8.31447/fuel_gas_class.molar_mass
            fuel_gamma = (fuel_gas_class.specific_heat)/(fuel_gas_class.specific_heat-fuel_R)
            fuel_rho = fuel_gas_class.density
            fuel_k = (2/(fuel_gamma+1))**((fuel_gamma+1)/(fuel_gamma-1))
            min_choked_p = 2 * 1.01325 / (2-fuel_gamma*fuel_k)
            if fuel_inj_p >= min_choked_p:
                fuel_mdot_start = self.fuel_core_CdA * np.sqrt(fuel_gamma*fuel_rho*fuel_inj_p*1e5*fuel_k)
                fuel_chokedstate = 'Choked'
            else:
                fuel_mdot_start = self.fuel_core_CdA * np.sqrt(2*(fuel_inj_p-1.01325) * 1e5 * fuel_rho)
        else:
            fuel_mdot_start = self.fuel_core_CdA * np.sqrt(2*(fuel_inj_p-1.01325) * 1e5 * fuel_rho)

        print(f'Total Start mdot: {(ox_mdot_start+fuel_mdot_start)*1e3:.4f} g/s')
        if ox_gas_class != None:
            print(f'Ox Start mdot: {ox_mdot_start*1e3:.4f} g/s ({ox_chokedstate})')
        else:
            print(f'Ox Start mdot: {ox_mdot_start*1e3:.4f} g/s')
        if fuel_gas_class != None:
            print(f'Fuel Start mdot: {fuel_mdot_start*1e3:.4f} g/s ({fuel_chokedstate})')
        else:
            print(f'Fuel Start mdot: {fuel_mdot_start*1e3:.4f} g/s')
        print(f'Start OF: {ox_mdot_start/fuel_mdot_start:.3f}')

class subplot:
    def __init__(self, yn, xn, title, engine):
        self.fig = plt.figure(constrained_layout=True)
        self.fig.suptitle(title)
        self.xn = xn
        self.yn = yn
        self.ax = {}
        self.ax2 = {}
        self.x = engine.x
        self.r = engine.r
        self.max_r = np.max([engine.rc, engine.re]) 

    def plt(self, loc, x, y, title, xlabel, ylabel, colour, draw_engine_contour=True, **label):
        if 'label' in label:
            label = label['label']
        else:
            label = None
        self.ax[loc] = self.fig.add_subplot(self.yn, self.xn, loc)
        self.ax[loc].plot(x, y, colour, label=label)
        self.ax[loc].set_title(title)
        self.ax[loc].set_xlabel(xlabel)
        self.ax[loc].set_ylabel(ylabel)
        self.ax[loc].grid(alpha=1)
        self.ax[loc].set_xlim(self.x[0]*1e3, self.x[-1]*1e3)
        self.ax[loc].set_aspect('equal', adjustable='datalim')
        self.ax[loc].xaxis.grid(AutoMinorLocator())
        self.ax[loc].yaxis.grid(AutoMinorLocator())
        if draw_engine_contour == True:
            self.ax2[loc] = self.ax[loc].twinx()
            self.ax2[loc].plot(self.x*1e3, self.r*1e3, color='gray')
            self.ax2[loc].set_ylim(0, self.max_r*5e3)

    def addline(self, loc, x, y, colour, label = None):
        self.ax[loc].plot(x, y, colour, label=label)
        self.ax[loc].legend()
        
class cea_fuel_water_mix:
    def __init__(self, alcohol, water_perc):
        self.alcohol = alcohol
        self.water_perc = water_perc
        if self.alcohol == 'Methanol':
            self.fuel_str = 'C 1 H 4 O 1'
        elif self.alcohol == 'Ethanol':
            self.fuel_str = 'C 2 H 6 O 1'
        card_str = f"""
        fuel {self.alcohol}   {self.fuel_str} 
        h,cal=-57040.0      t(k)=298.15       wt%={100-self.water_perc:.2f}
        oxid water H 2 O 1  wt%={self.water_perc:.2f}
        h,cal=-68308.  t(k)=298.15 rho,g/cc = 0.9998
        """
        add_new_fuel(f'{100-self.water_perc:.3g}% {self.alcohol} {self.water_perc:.3g}% Water', card_str)
    def str(self):
        return f'{100-self.water_perc:.3g}% {self.alcohol} {self.water_perc:.3g}% Water'

if __name__ == '__main__':
    system('cls')

    plt.ion()

    # hopper = engine('configs/hopperengine.cfg')
    # igniter = engine('configs/igniter.cfg')
    # csj = engine('configs/500N_csj.cfg')
    l9 = engine('configs/l9.cfg')

    pintle_inj = injector()
    pintle_inj.size_fuel_anulus(Cd = 0.8, ID = 17.3, OD = 18.06)
    pintle_inj.size_ox_holes(Cd = 0.509, d = 1.5, n = 60) #CD from NHNE
    # pintle_inj.set_ox_CdA(0.17e-5)
    pintle_inj.size_film_holes(Cd = 0.8, d = 1.5, n = 43)


    ambient_T = 24
    # igniter.eps = 1

    nitrous = Fluid(FluidsList.NitrousOxide)
    nitrous.update(Input.temperature(ambient_T), Input.quality(0))
    nitrous_vp = (nitrous.pressure-100)/1e5
    nitrous_density = nitrous.density

    # print(f'Nitrous Saturation Pressure: {nitrous_vp:.3f} bar')
    # print(f'Propane Saturation Pressure: {propane_vp:.3f} bar')

    fuel_p = 34.995
    nitrous_p = 34.520
    # vp = 32.5

    # nitrous.update(Input.pressure(vp*1e5), Input.quality(0))
    # T = nitrous.temperature
    # nitrous.update(Input.temperature(ambient_T), Input.pressure(nitrous_p*1e5))
    # nitrous.update(Input.temperature(T), Input.pressure(nitrous_p*1e5))
    # print(f"Nitrous Density: {nitrous.density:.3f} kg/m³")

    print(pintle_inj.fuel_A)

    l9.system_combustion_sim(
        fuel = 'Isopropanol',
        ox = 'N2O',
        fuel_upstream_p = fuel_p,
        ox_upstream_p = nitrous_p,
        fuel_rho = 790,
        ox_rho = 880,
        fuel_total_CdA = pintle_inj.fuel_total_CdA,
        ox_CdA = pintle_inj.ox_CdA,
        cstar_eff = 1.0,
        film_frac= pintle_inj.film_frac
    )
    l9.print_data()

    # OFsweep('Isopropanol', 'N2O', 0.5, 10, 25, 1, )

    plt.show(block=True)