# -*- coding: utf-8 -*-
import logging, sys
#Math tools
import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy.misc import derivative
from scipy.optimize import root, brentq, bisect, ridder
from mpmath import findroot
import matplotlib.pyplot as plt
from MathTools_pkg.MathTools import finding_SpecificGeometricPosition as find
#Therm utilities
from Thermo_pkg.Departure_pkg.References_Values import input_reference_values_function
from Thermo_pkg.Bubble_pkg.BubbleP import bubble_obj
from Thermo_pkg.Properties_pkg.Properties import Properties
from Thermo_pkg.Departure_pkg.EnthalpyEntropy import HSFv
from Thermo_pkg.ThermoTools_pkg import Tools_Convert
from Thermo_pkg.EOS_pkg.EOS_PengRobinson import PengRobinsonEos
from Thermo_pkg.Flash_pkg.FlashAlgorithm_main import getting_the_results_from_FlashAlgorithm_main as flash
#Data
from Data_pkg.PhysicalChemistryData_pkg.InputData___ReadThisFile import props
from Data_pkg.FlowData_pkg.input_flow_data import input_flow_data_function as FlowData
from Data_pkg.GeometricData_pkg.input_pipe_data import input_pipe_data_function as PipeData
from Data_pkg.GeometricData_pkg.input_pipe_data import areaVenturiPipe_function as Area
#Flow utilities
from Flow_pkg.FlowTools_file import FlowTools_class
from Flow_pkg.FlowTools_file import solution_concentration_set_function as conc
#Properties
# from CoolProp.CoolProp import PropsSI



'''
=================================================================================================================
TO LOGGING MSG(import logging and logging.basicConfig and logging.disable). 
1st option: write on the screen
2nd option: write on the file

To use this tool you must choose uncomment one of the following options below: the 1st OR the 2nd 
=================================================================================================================
'''
#=============
#[1st] OPTION:
meu_nivel_de_logging = logging.WARNING
logging.basicConfig(setLevel=meu_nivel_de_logging, format=' %(asctime)s - %(levelname)s - %(message)s')
#=============
#[2nd] OPTION
#logging.basicConfig(filename='Cleber_file.txt', level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
#=============
#[3] DISABEL: just uncomment the line below
# logging.disable(logging.WARNING)



R = 8314.34                         # J / (kmol K)



"""
#======================== INPUT PARAMETERS NOMENCLATURE (all international system unit) ======================
"""
#[1]================= DADOS DE ENTRADA ================= 
#
# mdotG: taxa (tx) massica de gás saturado [kg/s] {@ saturation}
# mdotF: tx massica de líquido saturado [kg/s] {@ saturation}
# mdotL_e: tx massica de líquido subresfriado entrada duto [kg/s] {@ subcooled liquid}
# MMixture: molecular weight - peso molecular da mistura (entrada duto) [kg/kmol]
# MM: vector with molar weight of each component "i" [kg/kmol], i.e., (MM = ([MMrefrig, MMpoe]))
# T_e: temperatura de entrada [K]
# p_e: pressao de entrada [Pa]
# pB: pressão de saturação [Pa] @ pB = pB(T_e)
# pB_v: é o próprio pB, porém na forma vetorial
# q_e: título de vapor entrada duto [-]
# mdotT = mdotG + mdotF: tx massica total (kg/s)
# densL, densF, densG: subcooled liquid density [kg/m3], satureted liquid density [kg/m3], satureted gas density [kg/m3]
# volL, volF, volG: specific volume subcooled liquid [m3/kg], specific volume satureted liquid [m3/kg],
#                         specific volume satureted gas [m3/kg]
# densL_e, spvolL_e: subcooled liquid density at duct entrance [kg/m3],
#                         specific volume subcooled liquid at duct entrance [m3/kg]
# volTP: specific volume two-phase [m3/kg]
# viscL: viscosidade do líquido subresfriado [kg/(m.s)]
# viscF: viscosidade do líquido saturado [kg/(m.s)] {@ saturation}
# visG: viscosidade do gás saturado [kg/(m.s)] {@ saturation}
# Gt: fluxo massico superficial total ((kg/s)/m2)
# u: flow speed [m/s]
# uG: satureted gas speed [m/s]
# uF: satureted liquid speed [m/s]
# Sr = uG/uF: speed ratio
# LC: light component (our refrigerant)
# Z: compressibility factor --> [p*spvolG = (Z*R*T) / MM]
# z: vector binary mixture molar composition at pipe's entrance: z = ([zR, zO])
# x: vector binary mixture molar composition at some pipe's position: x = ([xR, xO])
# f_D: Darcy friction factor
# f_F: Fanning friction factor
# hR_mass: reference enthalpy in mass base [J/kg]
# hR: referen ce enthalpy (in molar base) [J/kmol]
# sR_mass: reference entropy in mass base [J/kg K]
# sR: reference entropy (in molar base) [J/kmol K]
# CpL: subcooled liquid's specific heat [J/(kg K)] -- CpL = np.eisum('i,i', xRe, Cp)
# Cp: vector components' specific heat, which Cp1= Cp[0] and Cp2 = Cp[1]
# D: diameter for duct just (it isn't represent Venturi); it is a constant -- see Dvt  [m]
# l: any position at duct - [m]
# L: total length [m]
# pointsNumber: quantidade divisoes do duto, ou seja, quantidade de l steps [-]
# ks: absolute rugosity [m]
# rug = ks/D: relative roughness [-]
# angleVenturi_in: entrance venturi angle [rad]
# angleVenturi_out: outlet venturi angle [rad]  
# Dvt: venturi throat diameter where Dvt change according position, i.e, Dvt = Dvt(l) [m]
# liv: coordinate where venturi begins [m]
# lig: coordinate where venturi throat begins [m]
# lfg: coordinate where venturi throat ends [m]
# lfv: coordinate where venturi ends [m]
# Ac: cross section area; applied for entire circuit flow, i.e, Ac = Ac(l) [m2]
# rc: cross section radius [m]; applied for entire circuit flow, i.e, rc = rc(l) [m]
'''
=================================================================================
                                INPUT DATA

[1] - TAKING SPECIFIC HEAT FROM props (file InputData___ReadThisFile.py)

[2] - Reference values of temperature, enthalpy and entropy are collected from 
  ...input_reference_values_function() (file Reference_Values.py)
[3] - FLOW DATA is from input_flow_data_function() (file input_flow_data_function.py)
[4] - DUCT DATA is from input_pipe_data_function() (finle input_pipe_data_function.py)
=================================================================================
'''
#Global variables
(pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props
(TR, hR_mass, sR_mass) = input_reference_values_function()
(p_e, T_e, mdotL_e, fmR134a) = FlowData() # <=============================== change here
(angleVenturi_in, angleVenturi_out, ks, L, D, Dvt, liv, lig, lfg, lfv) = PipeData() # <=============================== change here


'''
=================================================================================================================
MODELS FOR DENSITY/VISCOSITY/FRICTION/TWO-PHASE
=================================================================================================================
'''
density_models = {'_jpDias':'jpDias', '_ELV':'ELV'}
viscosity_models = {'_jpDias':'jpDias', '_NISSAN':'NISSAN'}
friction_models = {'_Colebrook':'Colebrook', '_Churchill':'Churchill'}
viscosity_TP_models = {'_McAdams':'McAdams', '_Cicchitti':'Cicchitti', '_Dukler':'Dukler'}


'''
=================================================================================================================
CREATING NECESSARY OBJECT
=================================================================================================================
'''
prop_obj = Properties(pC, Tc, AcF, omega_a, omega_b, kij)
eos_obj = PengRobinsonEos(pC, Tc, AcF, omega_a, omega_b, kij)



'''
=================================================================================================================
GETTING MOLAR AND MASS VECTOR CONCENTRATION
=================================================================================================================
'''

try:
    z, z_mass = conc(fmR134a, MM, 'mass')
    #There is a call for 'err', which it was built in conc function
except Exception as err:
    print('Error when try to build concentration z and z_mass  => ' + str(err))


'''
=================================================================================================================
GETTING BUBBLE PRESSURE, MOLAR MIXTURE WEIGHT and REFERENCES ENTHALPY/ENTROPY 
=================================================================================================================
'''
def saturationPressure_ResidualProperties_MolarMixture_function():
    '''
    Callling some thermodynamics variables \n
    All of them are objects 
    '''
    MMixture = prop_obj.calculate_weight_molar_mixture(MM, z, "saturated_liquid")
    hR = hR_mass * MMixture
    sR = sR_mass * MMixture
    pB = bubble_obj(T_e, z)[0]
    pR = bubble_obj(TR, z)[0]

    return (hR, sR, pB, pR, MMixture)

(hR, sR, pB, pR, MMixture) = saturationPressure_ResidualProperties_MolarMixture_function()


logging.warning('pB = ' + str(pB))

'''
=================================================================================================================
CREATING MORE NECESSARY OBJECTS
=================================================================================================================
'''

hsFv_obj = HSFv(pC, TR, Tc, AcF, Cp, MM, hR, sR) #to obtain enthalpy

FlowTools_obj = FlowTools_class(pC, Tc, AcF, omega_a, omega_b, kij, mdotL_e)






'''
==================================================================================================================
'''

# if __name__== '__main__':
#     print('\n---------------------------------------------------')
#     print('[1] - Guess pB [Pa]= %.8e' % pG)
#     print('[2] - ======> at T = %.2f [C], pB = %.8e [Pa] ' % ((T_e - 273.15), pB) + '\n')
#     print('[3] - Concentration vapor phase [molar] = ', y.round(3))
#     print('[4] - Concentration vapor phase [mass] = ', y_mass.round(3))
#     print('[5] - Pay attention if Sy is close to unity (Sy = %.10f) [molar]' % Sy)
#     print('[6] - Feed global {mass} fraction = ', xRe_mass.round(3))
#     print('[7] - Feed global {molar} fraction = ', xRe.round(3))
#     print()




'''
=================================================================================================================
BUILDING INITIAL VALUES
=================================================================================================================
'''

def initial_values():
    '''
    Necessary set initial values to get in solve_ivp numerical method \n
    Return: u_e, p_e, h_e
    '''
    Ac_e = Area(0.0) 
    volL_e = FlowTools_obj.specificVolumeLiquid_Wrap(p_e, T_e, MM, z, density_models['_jpDias'])
    u_e = (mdotL_e * volL_e) / Ac_e                  # Para obter u_e (subcooled liquid)
    _F_V, h_e, _s = hsFv_obj(p_e, T_e, z)            #Para obter h_e (subcooled liquid, so F_V = 0)
    return (u_e, p_e, h_e)

(u_e, p_e, h_e) = initial_values()



#[6]============================ MAIN - CODE =========================================
#packing the parameters to get in systemEDOsinglePhase()
singlePhaseModels = {'density':density_models['_jpDias'], 'viscosity':viscosity_models['_NISSAN'],
          'friction':friction_models['_Colebrook']}

#ODE system
def edo_sp(l, uph):
    '''
    Target: resolver um sistema de EDO's em l para u, p e h com linalg.solve; \n
        [1] - Input: \n
            [1.a] vetor comprimento circuito (l) \n
            [1.b] vazão mássica líquido subresfriado entrada (mdotL_e) \n
            [1.c] diâmetro tubo (D) \n
            [1.d] peso molecular de cada componente (MM) \n
            [1.e] viscosidade da líquido (viscL) \n
            [1.f] rugosidade absoluta (ks) \n
            [1.g] entalpia entrada duto (h_e) \n
            [1.h] temperatura entrada duto (T_e) \n
            [1.i] vetor calor específico (Cp = ([CpR, CpO])) \n
            [1.j] vetor concentração molar entrada duto (  z = ([zR, zO])  ) \n
            [1.l] vetor concentração massica entrada duto (  z_mass = ([zR_mass, zO_mass])  ) \n
            [1.m] incremento/passo no comprimento l do duto (incril); saída adicional do np.linspace \n
            [1.n] uma lista com parâmetros do escoamento (flow_list) \n
            [1.o] uma lista com os parametros geométricos do circuito escoamento (geometric_list) \n
        [2] - Return: (du/dl, dp/dl, dh/dl)
    '''
    ''''''
    # unpack
    u, p, h = uph

    # calculating CpL, radius, Gt, area
    CpL = FlowTools_obj.specificLiquidHeat_jpDias(T_e, p, z_mass)
    T = T_e + (h - h_e) / CpL
    Ac_e = Area(0.0)
    Ac = Ac_e #Area(0.0)
    rc = np.sqrt(Ac / np.pi)
    Dc = 2. * rc
    Gt = mdotL_e / Ac
    Gt2 = np.power(Gt, 2)
    volL = FlowTools_obj.specificVolumeLiquid_Wrap(p, T, MM, z, singlePhaseModels['density'])
    
    # area average
    dl = 1e-5
    Ac_e = Area(0.0)
    avrg_A = Ac_e #(AreaAc_e #(l) + A Ac_e = Area(0.0rea(l + dl)) / 2. # Be careful! 1e-5 it's the same value used to 'h' in dfdx
    dAdl = 0.0 #derivative(Area,Ac_e # l, dl)
    
    # calculating beta
    def volL_func(T, p, MM, z, density=singlePhaseModels['density']):
        '''
        This function has been created to make possible evaluate dvolLdT \n
        Where dvolLdT is the derivative of specific liquid volume with temperature \n

        Return: volL
        '''
        return FlowTools_obj.specificVolumeLiquid_Wrap(p, T, MM, z, density)
    dT = 1e-5

    dvolLdT = derivative(volL_func, T, dT, args=(p, MM, z))
    beta = np.power(volL, -1) * dvolLdT


    # friction factor
    viscL = FlowTools_obj.viscosityLiquid_Wrap(p, T, z, z_mass, singlePhaseModels['viscosity'])
    Re_mon = FlowTools_obj.reynolds_function(Gt, Dc, viscL)
    f_F = FlowTools_obj.frictionFactorFanning_Wrap(Re_mon, ks, Dc, singlePhaseModels['friction'])
    
    # setting the matrix coefficients
    A11, A12, A13 = np.power(u,-1), 0., (- beta / CpL)     
    A21, A22, A23 = u, volL, 0.
    A31, A32, A33 = u, 0., 1.

    aux = -2.0 * Gt2 * np.power(volL, 2) * (f_F / Dc )
    C1, C2, C3 = (- dAdl / avrg_A), aux, aux

    matrizA = np.array([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])
    RHS_C = np.array([C1, C2, C3])
    dudl, dpdl, dhdl = np.linalg.solve(matrizA, RHS_C)

    return (dudl, dpdl, dhdl)


def saturation_point_met(l, uph):
    _u, p, h = uph
    CpL = FlowTools_obj.specificLiquidHeat_jpDias(T_e, p, z_mass)
    T = T_e + (h - h_e) / CpL

    if (p - 0.97 * pB) < 0.0:
        q, is_stable, K_values_newton, initial_K_values = flash(p, T, pC, Tc, AcF, z)
        if is_stable is False: return 0.0
        else: return 1.
    else: return 1.



def main():
    pointsNumber = 2000
    l, incril = np.linspace(0, L, pointsNumber + 1, retstep=True)
    # l_crit = np.array([liv, lig, lfg, lfv])
    saturation_point_met.terminal = True
    saturation_point_met.direction = 0
    return solve_ivp(edo_sp, [0.0, L], [u_e, p_e, h_e], method='Radau', max_step=incril, events=saturation_point_met)


# UNPACKING THE RESULTS
uph_sp = main()
u_sp = uph_sp.y[0,:]
p_sp = uph_sp.y[1,:]
h_sp = uph_sp.y[2,:]
l_sp = uph_sp.t
pB_v = pB * np.ones_like(u_sp)


logging.warning('temperatura entrada duto T_e = ' + str(T_e))
logging.warning('flashing point = ' + str(uph_sp.t_events))
logging.warning('última pressao = ' + str(p_sp[-1]))




'''
===============&&&&&&&&&&&&&&&&&&&===========++++++++++++++===================
'''
# sys.exit(0)

twoPhase_models = {'density':density_models['_jpDias'], 'viscosity':viscosity_models['_NISSAN'],
          'friction':friction_models['_Colebrook'],
                 'viscosityTP':viscosity_TP_models['_McAdams']}

# ODE system for Two Phase Flow
def edo_2p(l, uph):

    ''''''
    # unpack
    u, p, h = uph
    Ac_e = Area(0.0)
    Ac = Ac_e #Area(l)
    rc = np.sqrt(Ac / np.pi)
    Dc = 2. * rc
    Gt = mdotL_e / Ac
    Gt2 = np.power(Gt, 2)

    # ======================================================================
    #                      find(T)                                         #
    # ======================================================================
    LI = 0.95 * T_e
    LS = 1.05 * T_e
    def find_temperature(T, p, z, h):
        _F_v, hELV, _sELV = hsFv_obj(p, T, z)
        return (hELV - h)

    try:
        # T, converged = brentq(find_temperature, LI, LS, args=(p, z, h), xtol=1e-6, full_output=True)
        T, converged = ridder(find_temperature, LI, LS, args=(p, z, h), xtol=1e-3, full_output=True)
        if converged is False:
            raise Exception('Not converged' + str(converged))
    except Exception as msg_err:
        print('It has been difficult to find the roots ' + str(msg_err))

    try:
        q, is_stable, K_values_newton, initial_K_values = flash(p, T, pC, Tc, AcF, z)
        if is_stable is False:
            x = z / (q * (K_values_newton - 1.) + 1.)
            y = K_values_newton * x
        else:
            q, y, x = 0.0, np.zeros_like(z), z
            raise Exception(' Quality q = ' + str(q))
    except Exception as new_err:
        print('This mixture is stable yet! (q < 0.0)! Artificially q and y are set to ZERO ' + str(new_err))


    logging.warning('(T = %s ) and (P = %s ) at l = %s ' % (T, p, l))
    logging.warning('Vapor quality = ' + str(q))
    x_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, x)

    # ======================================================================
    #                            area derivative                           #
    # ======================================================================
    dl = 1e-5
    avrg_A = Ac_e #(Area(l) + Area(l + dl)) / 2.
    dAdl = 0.0 #derivative(Area, l, dl)
    # ======================================================================
    #                      compressibility                                 #
    # ======================================================================
    def volTP_func(p, T, MM, x, y, density=twoPhase_models['density']):
        '''
        This function has been created to evaluate compressibility = (1 / volTP) * dvolTPdp \n
        Obs: negative signal is omitted here, but it is used ahead \n

        Return: volTP
        '''
        volL = FlowTools_obj.specificVolumeLiquid_Wrap(p, T, MM, x, density)
        volG = FlowTools_obj.specificVolumGas(p, T, MM, y)
        volTP = FlowTools_obj.specificVolumeTwoPhase(q, volG, volL)
        return volTP

    dp = 1e-5
    avrg_volTP = (volTP_func(p, T, MM, x, y) + volTP_func(p + dp, T, MM, x, y)) / 2.
    dvolTPdp = derivative(volTP_func, p, dp, args=(T, MM, x, y))
    compressibility = np.power(avrg_volTP, -1) * dvolTPdp
    # =====================================================================
    #            calculating dvolTPdh (eq. A.13 da Tese)                  #
    # =====================================================================


    def volL_func(x, p, T, MM, density=twoPhase_models['density']):
        '''
        This function has been created to make possible evaluate dvolTPdh \n
        where dvolTPdh is the derivative of specific two-phase volume with enthalpy \n

        Return: volL
        '''
        volL = FlowTools_obj.specificVolumeLiquid_Wrap(p, T, MM, x, density)

        return volL

    dx = 1e-5
    dvolLdxR = derivative(volL_func, x, dx, args=(p, T, MM))

    def hL_func(x, z, p, T, pR, TR, hR, Cp, MM, fluid_type='saturated_liquid'):
        '''
        This function has been created to make possible evaluate dvolTPdh \n
        where dvolTPdh is the derivative of specific two-phase volume with enthalpy \n

        Return: hL
        '''
        H_L = prop_obj.calculate_enthalpy(TR, T, pR, p, x, z, hR, Cp, fluid_type)
        M_L = prop_obj.calculate_weight_molar_mixture(MM, x, fluid_type)
        hL = H_L / M_L
        return hL

    dhLdxR = derivative(hL_func, x, dx, args=(z, p, T, pR, TR, hR, Cp, MM))

    volL = FlowTools_obj.specificVolumeLiquid_Wrap(p, T, MM, x, twoPhase_models['density'])
    volG = FlowTools_obj.specificVolumGas(p, T, MM, y)
    H_G = prop_obj.calculate_enthalpy(TR, T, pB, p, y, z, hR, Cp, 'saturated_vapor')
    M_V = prop_obj.calculate_weight_molar_mixture(MM, y, 'saturated_vapor')
    if y.all() == 0.0: hG = 0.0
    else: hG = H_G / M_V
    H_L = prop_obj.calculate_enthalpy(TR, T, pB, p, x, z, hR, Cp, 'saturated_liquid')
    M_L = prop_obj.calculate_weight_molar_mixture(MM, x, 'saturated_liquid')
    hL = H_L / M_L
    vol_fg = volG - volL
    h_fg = hG - hL
    dvolTPdh = (vol_fg - (1. - x[0]) * dvolLdxR) / (h_fg - (1. - x[0]) * dhLdxR)
    # ======================================================================
    #                      two-phase multiplier                            #
    # ======================================================================
    viscL_fo = FlowTools_obj.viscosityLiquid_Wrap(p, T, z, z_mass, twoPhase_models['viscosity'])
    volL_fo = FlowTools_obj.specificVolumeLiquid_Wrap(p, T, MM, z,twoPhase_models['density'])
    viscL = FlowTools_obj.viscosityLiquid_Wrap(p, T, x, x_mass, twoPhase_models['viscosity'])
    volTP = FlowTools_obj.specificVolumeTwoPhase(q, volG, volL)
    viscTP = FlowTools_obj.viscosityTwoPhase(q, volG, volTP, viscL,twoPhase_models['viscosityTP'])
    phiLO2 = FlowTools_obj.twoPhaseMultiplier(q, viscTP, viscL, volG, volL)
    # ======================================================================
    #                      friction factor f_LO                            #
    # ======================================================================
    Re_mon = FlowTools_obj.reynolds_function(Gt, Dc, viscL_fo)
    f_FLO = FlowTools_obj.frictionFactorFanning_Wrap(Re_mon, ks, Dc, twoPhase_models['friction'])
    # ======================================================================
    #                      setting the matrix coefficients                 #
    # ======================================================================
    A11, A12, A13 = np.power(u, -1), - compressibility, - (dvolTPdh / volTP)
    A21, A22, A23 = u, volTP, 0.
    A31, A32, A33 = u, 0., 1.

    aux = - volTP * phiLO2 * (2.0 * Gt2 * volL_fo * (f_FLO / Dc))
    C1, C2, C3 = (- dAdl / avrg_A), aux, aux
    # ======================================================================
    #                      matrix solving                                  #
    # ======================================================================
    matrizA = np.array([[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]])
    RHS_C = np.array([C1, C2, C3])
    dudl, dpdl, dhdl = np.linalg.solve(matrizA, RHS_C)

    return (dudl, dpdl, dhdl)


#Executing two-phase
l_sp_0 = l_sp[-1]
u_sp_0 = u_sp[-1]
p_sp_0 = p_sp[-1]
h_sp_0 = h_sp[-1]
tolerance = np.array([1e-2, 1e-1, 1e-1])
def main2():
    pointsNumber = 700
    l, incril = np.linspace(l_sp_0, L, pointsNumber + 1, retstep=True)
    return solve_ivp(edo_2p, [l_sp_0, L], [u_sp_0, p_sp_0, h_sp_0], method='Radau', max_step=incril, atol=tolerance)



# SOLVING - INTEGRATING

uph_tp = main2()
# UNPACKING THE RESULTS
u_tp = uph_tp.y[0,:]
p_tp = uph_tp.y[1,:]
h_tp = uph_tp.y[2,:]
l_tp = uph_tp.t



#Preparing data to plot
u_p = np.concatenate((u_sp, u_tp), axis=0)
p_p = np.concatenate((p_sp, p_tp), axis=0)
h_p = np.concatenate((h_sp, h_tp), axis=0)
l_p = np.concatenate((l_sp, l_tp), axis=0)














#
# # #[9]=========================== PLOTTING =====================
#
# CpL = FlowTools_obj.specificLiquidHeat_jpDias(T_e, p_e, z_mass)
# T = T_e + (h - h_e) / CpL
#
# index_l_150mm = find(l, 0.150)
# # print('taking index of vector where l = 150mm', index_l_150mm)
#
# index_l_640mm = find(l, 0.640)
# # print('taking index of vector where l = 640mm', index_l_640mm)
# pinter = p[index_l_640mm]
# hinter = h[index_l_640mm]
# Tinter = T_e + (hinter - h_e) / CpL
# # print('quem é meu z? ', z)
#
#
# volL_inter = FlowTools_obj.specificVolumeLiquid_Wrap(pinter, Tinter, MM, z, z_mass, density_models['_jpDias'])
# densL_inter = 1. / volL_inter
# print('olha a densidade do ELV', densL_inter)
# print('veja como é meu z = ', z, 'e também z_mass', z_mass)
# Acinter = Area(0.640)
# Gtinter = mdotL_e / Acinter
# rinho = np.linspace(-D / 2, D / 2, 101)
#
# # q = 0.4
# # viscF = FlowTools_obj.viscosityLiquid_Wrap(p, T, z, z_mass, 'jpDias')
# # volG = FlowTools_obj.specificVolumGas(p, T, MM, z)
# # volF = FlowTools_obj.specificVolumeLiquid_Wrap(p, T, MM, z, z_mass, density_model='jpDias')
# # vol2P = FlowTools_obj.specificVolumeTwoPhase(q, volG, volF)
# # visc2P = FlowTools_obj.viscosityTwoPhase(q,volG, vol2P, viscF, viscTP_model='Dukler')
# # logging.warning('viscosidade bifásica deu certo?' + str(visc2P))
#
# R = D / 2.
# ur = np.zeros_like(rinho)
# for i, r in enumerate(rinho):
#     ur[i] = 2 * Gtinter * volL_inter * (1. - (r / R) ** 2)
#
plt.figure(figsize=(7,5))
# plt.xlim(0.0,1.6)
# plt.ylim(-8.0, 8.0)
plt.xlabel('l [mm]')
plt.ylabel('u [m/s]')
# plt.plot(ur, rinho * 1e3)
plt.plot(l_p, u_p)
plt.legend(['$u_{incompressivel}$ ao longo do duto'], loc=1)
#
#
plt.figure(figsize=(7,5))
# plt.xlim(0.0,1.6)
# plt.ylim(-8.0, 8.0)
plt.xlabel('l [mm]')
plt.ylabel('p [Pa]')
# plt.plot(ur, rinho * 1e3)
plt.plot(l_p, p_p)
plt.legend(['pressao ao longo do duto'], loc=1)

plt.figure(figsize=(7,5))
# plt.xlim(0.0,1.6)
# plt.ylim(-8.0, 8.0)
plt.xlabel('l [mm]')
plt.ylabel('h [J/kgK]')
# plt.plot(ur, rinho * 1e3)
plt.plot(l_p, h_p)
plt.legend(['entalpia ao longo do duto'], loc=1)
#
# # plt.figure(figsize=(7,5))
# # #plt.ylim(20,120)
# # plt.xlabel('l [m]')
# # plt.ylabel('T [K]')
# # plt.plot(l, T)
# # plt.legend(['Temperatura Esc. Incompressivel'], loc=3)
#
#
# plt.figure(figsize=(7,5))
# #plt.xlim(0.6,0.8)
# plt.xlabel('l [m]')
# plt.ylabel('u [m/s]')
# plt.plot(l, u)
# plt.legend(['$u_{incompressivel}$ ao longo do duto'], loc=1) #loc=2 vai para canto sup esq
#
#
# plt.figure(figsize=(7, 5))
# plt.title('Grafico comparativo com pg 81 Tese Dalton')
# plt.grid(True)
# plt.xlabel('$l$* [m]')
# plt.ylabel('(p - p#1) [Pascal]')
# # plt.plot((l[(point_l150mm-1):] - 0.150), (p[(point_l150mm - 1):] - p150mm))
# plt.plot((l[index_l_150mm:] - 0.150), (p[index_l_150mm:] - p[index_l_150mm]))
# plt.xlim(0.0, 0.9)
# # plt.ylim(8e5, 10e5)
# # plt.legend(['Pressao Esc. Incompressivel'], loc=3)
# # plt.plot(l, pB_v)
# # plt.legend(['Pressao Esc. Incompressivel', 'Pressão Saturação'], loc=3)
# plt.legend('Pressão Esc. Incompressível', loc=1)
#
# # plt.figure(figsize=(7,5))
# # #plt.ylim(20,120)
# # plt.xlabel('l [m]')
# # plt.ylabel('H [J/kg]')
# # plt.plot(l, h)
# # plt.legend(['$h_{incompressivel}$ ao longo do duto'], loc=3)
#
#
plt.show()
plt.close('all')
#
