# -*- coding: utf-8 -*-
import logging, sys, cProfile
# Math tools
import numpy as np
import pandas as pd
from scipy.integrate import odeint, solve_ivp
from scipy.misc import derivative
from scipy.optimize import root, brentq, bisect, ridder
from mpmath import findroot
import matplotlib.pyplot as plt
from MathTools_pkg.MathTools import finding_SpecificGeometricPosition as find
# Therm utilities
from Thermo_pkg.Departure_pkg.References_Values import input_reference_values_function
from Thermo_pkg.Bubble_pkg.BubbleP import bubble_obj
from Thermo_pkg.Properties_pkg.Properties import Properties
from Thermo_pkg.Departure_pkg.EnthalpyEntropy import HSFv
from Thermo_pkg.ThermoTools_pkg import Tools_Convert
from Thermo_pkg.EOS_pkg.EOS_PengRobinson import PengRobinsonEos
from Thermo_pkg.Flash_pkg.FlashAlgorithm_main import getting_the_results_from_FlashAlgorithm_main as flash
# Data
from Data_pkg.PhysicalChemistryData_pkg.InputData___ReadThisFile import props
from Data_pkg.FlowData_pkg.input_flow_data import input_flow_data_function as FlowData
from Data_pkg.GeometricData_pkg.input_pipe_data import input_pipe_data_function as PipeData
from Data_pkg.GeometricData_pkg.input_pipe_data import areaVenturiPipe_function as Area
# Flow utilities
from Flow_pkg.FlowTools_file import FlowTools_class
from Flow_pkg.FlowTools_file import solution_concentration_set_function as conc
#Properties
# from CoolProp.CoolProp import PropsSI
# Find who is consuming more time




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
(p_e, T_e, mdotL_e, fmR134a) = FlowData()
(angleVenturi_in, angleVenturi_out, ks, L, D, Dvt, liv, lig, lfg, lfv) = PipeData()


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
    # There is a call for 'err', which it was built in conc function
except Exception as err:
    print('Error when try to build concentration z and z_mass  => ' + str(err))


'''
=================================================================================================================
GETTING BUBBLE PRESSURE, MOLAR MIXTURE WEIGHT and REFERENCES ENTHALPY/ENTROPY 
=================================================================================================================
'''


def saturationpressure_residualproperties_molarmixture_function():
    '''
    Calling some thermodynamics variables \n
    All of them are objects \n

    hR: it is the molar enthalpy reference [J/kmol] based on mass enthalpy reference \n
    sR: it is the molar entropy reference [J/kmolK] based on mass entropy reference \n
    TR: reference temperature [K] \n
    pB: bubble pressure calculated based on entrance temperature [Pa] \n
    pR: reference pressure, which is the bubble pressure calculated at TR with
     feed molar composition (z) [Pa] \n
    '''
    MMixture = prop_obj.calculate_weight_molar_mixture(MM, z, "saturated_liquid")
    hR = hR_mass * MMixture
    sR = sR_mass * MMixture
    pB_e = bubble_obj(T_e, z)[0]
    pR = bubble_obj(TR, z)[0]

    return hR, sR, pB_e, pR, MMixture


(hR, sR, pB_e, pR, MMixture) = saturationpressure_residualproperties_molarmixture_function()

logging.warning('Bubble pressure evaluated at pB = pB(T_e, z) --> ' + str(pB_e))

'''
=================================================================================================================
CREATING MORE NECESSARY OBJECTS
=================================================================================================================
'''

hsFv_obj = HSFv(pC, TR, Tc, AcF, Cp, MM, hR, sR) #to obtain enthalpy

flowtools_obj = FlowTools_class(pC, Tc, AcF, omega_a, omega_b, kij, mdotL_e)






'''
==================================================================================================================
'''

# if __name__== '__main__':
#     print('\n---------------------------------------------------')
#     print('[1] - Guess pB_e [Pa]= %.8e' % pG)
#     print('[2] - ======> at T = %.2f [C], pB_e = %.8e [Pa] ' % ((T_e - 273.15), pB_e) + '\n')
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
    volL_e = flowtools_obj.specificVolumeLiquid_Wrap(p_e, T_e, MM, z, density_models['_jpDias'])
    u_e = (mdotL_e * volL_e) / Ac_e                  # Para obter u_e (subcooled liquid)
    _q, h_e, _s = hsFv_obj(p_e, T_e, z)            #Para obter h_e (subcooled liquid, so F_V = 0)
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
    global T_e, z, z_mass, ks, mdotL_e, MM
    # unpack

    u, p, h = uph

    # calculating CpL, radius, Gt, area
    CpL = flowtools_obj.specificLiquidHeat_jpDias(T_e, p, z_mass)
    T = T_e + (h - h_e) / CpL
    Ac = Area(l)
    rc = np.sqrt(Ac / np.pi)
    Dc = 2. * rc
    Gt = mdotL_e / Ac
    Gt2 = np.power(Gt, 2)
    volL = flowtools_obj.specificVolumeLiquid_Wrap(p, T, MM, z, singlePhaseModels['density'])
    
    # area average
    dl = 1e-5
    avrg_A = (Ac + Area(l + dl)) / 2.
    dAdl = derivative(Area, l, dl)

    # LOOK HERE HOW I WAS WRITING ===> density=singlePhaseModels['density']
    # calculating beta
    def volL_func(temperature, pressure, molar_weight, molar_composition, density=singlePhaseModels['density']):
        '''
        This function has been created to make possible evaluate dvolLdT \n
        Where dvolLdT is the derivative of specific liquid volume with temperature \n

        Return: volL
        '''
        volL = flowtools_obj.specificVolumeLiquid_Wrap(pressure, temperature, molar_weight,
                                                       molar_composition, density)
        return volL


    dT = 1e-5
    dvolLdT = derivative(volL_func, T, dT, args=(p, MM, z))
    beta = np.power(volL, -1) * dvolLdT


    # friction factor
    viscL = flowtools_obj.viscosityLiquid_Wrap(p, T, z, z_mass, singlePhaseModels['viscosity'])
    # logging.warning('viscosity = %s and density = %s' % (viscL * 1e6, (1. / volL)))
    Re_mon = flowtools_obj.reynolds_function(Gt, Dc, viscL)
    # logging.warning('(Re_mon = %s), na posicao (l = %s)' % (Re_mon, l))
    f_F = flowtools_obj.frictionFactorFanning_Wrap(Re_mon, ks, Dc, singlePhaseModels['friction'])
    
    # setting the matrix coefficients
    A11, A12, A13 = np.power(u,-1), 0., (- beta / CpL)     
    A21, A22, A23 = u, volL, 0.
    A31, A32, A33 = u, 0., 1.

    aux = -2.0 * Gt2 * np.power(volL, 2) * (f_F / Dc )
    C1, C2, C3 = (- dAdl / avrg_A), aux, aux

    matrizA = np.array([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])
    RHS_C = np.array([C1, C2, C3])
    dudl, dpdl, dhdl = np.linalg.solve(matrizA, RHS_C) #flinha.shape is (3,)

    return dudl, dpdl, dhdl


def saturation_point_met(l, uph):
    '''
    This function it is necessary to identify the flash point, i.e., the initial phase change
    :param t: independent variable \n
    :param uph: dependent variables (packed), i.e., uph = uph(t) \n

    :return: 0.0 or 1.0 (must be a float)
    '''
    _u, p, h = uph
    CpL = flowtools_obj.specificLiquidHeat_jpDias(T_e, p, z_mass)
    T = T_e + (h - h_e) / CpL

    if (p - 0.97 * pB_e) < 0.0:
        q, is_stable, K_values_newton, initial_K_values = flash(p, T, pC, Tc, AcF, z)
        if is_stable is False: return 0.0
        else: return 1.
    else: return 1.



def main():
    saturation_point_met.terminal = True
    saturation_point_met.direction = 0
    uphINI = np.array([u_e, p_e, h_e])
    return solve_ivp(edo_sp, [0.0, L], uphINI, method='BDF', max_step=1e-2,
                     events=saturation_point_met, vectorized=False)


# =====================================================================================
# ------------------------------EXECUTING main(single-phase)--------------------------
uph_sp = main()
# =====================================================================================
# UNPACKING THE RESULTS
u_sp = uph_sp.y[0,:]
p_sp = uph_sp.y[1,:]
h_sp = uph_sp.y[2,:]
l_sp = uph_sp.t
pB_e_v = pB_e * np.ones_like(u_sp)


logging.warning('temperatura entrada duto T_e = ' + str(T_e))
logging.warning('velocidade (u = %s ) na posicao (l = %s )  ' % (u_sp[3], l_sp[3]))
logging.warning('velocidade (u = %s ) na posicao (l = %s )  ' % (u_sp[6], l_sp[6]))
logging.warning('flashing point = ' + str(uph_sp.t_events))
logging.warning('última pressao = ' + str(p_sp[-1]))
logging.warning('velocidade no flash point = ' + str(u_sp[-1]))


# Single-phase results: Taking the transpose (transform to a vector column)
l_singlephase = uph_sp.t.T
size = l_singlephase.shape[0]
# Forcing vector length to become two dimensional
l_singlephase.shape = (size, 1)
# Taking the transpose
uph_singlephase = uph_sp.y.T
# Concatenating time with uph_sp
tuph_singlephase = np.append(l_singlephase, uph_singlephase, axis=1)
# Send matrix to pandas
df_sp = pd.DataFrame(tuph_singlephase,columns=['length(m)', 'velocity(m/s)', 'pressure(Pa)', 'enthalpy(J/kgK)'])
single_phase_case = 'single_phase'
# r na frente é devido a problematica da barra invertida do windows
df_sp.to_csv(r'../Results_pkg/{}.csv'.format(single_phase_case), sep='\t', float_format='%.4f')

# /Users/Adm/Documents/aa.UniversUFSC/Tese_Doutorado/Git_Reposit_Local_Mac/FlashingProjectGitHub



'''
==============================================================
= NEXT WE HAVE THE MAIN SCRIPT FOR THE TWO-PHASE FLOW        =
==============================================================
'''
# sys.exit(0)

twoPhase_models = {'density':density_models['_jpDias'], 'viscosity':viscosity_models['_NISSAN'],
          'friction':friction_models['_Colebrook'],
                 'viscosityTP':viscosity_TP_models['_McAdams']}

# ODE system for Two Phase Flow
def edo_2p(l, uph):

    '''
    :param t: independent variable \n
    :param uph: dependent variables (packed), i.e., uph = uph(t) \n

    :return: EDO system with derivative of uph with dt, i.e., d(uph)dt
    '''
    global T_e, z, z_mass, ks, mdotL_e, MM, pR, TR, hR
    # unpack
    u, p, h = uph
    Ac = Area(l)
    rc = np.sqrt(Ac / np.pi)
    Dc = 2. * rc
    Gt = mdotL_e / Ac
    Gt2 = np.power(Gt, 2)

    # ======================================================================
    #                      find(T)                                         #
    # ======================================================================
    TI = 0.9 * T_e
    TS = 1.1 * T_e

    def find_temperature(temperature, pressure, molar_composition, enthalpy):
        _q, helv, _s = hsFv_obj(pressure, temperature, molar_composition)
        return helv - enthalpy

    try:
        T, converged = ridder(find_temperature, TI, TS, args=(p, z, h), xtol=1e-3, full_output=True)
        if converged is False:
            raise Exception('Not converged' + str(converged))
    except Exception as msg_err:
        print('It has been difficult to find the roots ' + str(msg_err))

    try:
        q, is_stable, K_values_newton, _initial_K_values = flash(p, T, pC, Tc, AcF, z)
        if is_stable is False:
            x = z / (q * (K_values_newton - 1.) + 1.)
            y = K_values_newton * x
        else:
            q, y, x = 0.0, np.zeros_like(z), z
            raise Exception(' Quality q = ' + str(q))
    except Exception as new_err:
        print('This mixture is stable yet! (q < 0.0)! Artificially q and y are set to ZERO ' + str(new_err))

    logging.warning('(T = %s ), (P = %s ), (i = %s) and (u = %s ) at l = %s ' % (T, p, h, u, l))
    logging.warning('Vapor quality = ' + str(q))

    x_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, x)

    # ======================================================================
    #                            area derivative                           #
    # ======================================================================
    dl = 1e-5
    avrg_A = (Ac + Area(l + dl)) / 2.
    dAdl = derivative(Area, l, dl)
    # ======================================================================
    #                      compressibility                                 #
    # ======================================================================
    def volTP_func(pressure, temperature, quality, molar_weight, liquid_molar_composition, vapor_molar_composition,
                   density=twoPhase_models['density']):
        '''
        This function has been created to evaluate compressibility = (1 / volTP) * dvolTPdp \n
        Obs: negative signal is omitted here, but it is used ahead \n

        Return: volTP
        '''
        volL_local = flowtools_obj.specificVolumeLiquid_Wrap(pressure, temperature, molar_weight,
                                                       liquid_molar_composition, density)
        volG_local = flowtools_obj.specificVolumGas(pressure, temperature, molar_weight, vapor_molar_composition)
        volTP = flowtools_obj.specificVolumeTwoPhase(quality, volG_local, volL_local)
        return volTP

    dp = 1e-5
    avrg_volTP = (volTP_func(p, T, q, MM, x, y) + volTP_func(p + dp, T, q, MM, x, y)) / 2.
    dvolTPdp = derivative(volTP_func, p, dp, args=(T, q, MM, x, y))
    compressibility = np.power(avrg_volTP, -1) * dvolTPdp
    # =====================================================================
    #            calculating dvolTPdh (eq. A.13 da Tese)                  #
    # =====================================================================

    def volL_func(liquid_molar_composition, pressure, temperature, molar_weight, density=twoPhase_models['density']):
        '''
        This function has been created to make possible evaluate dvolTPdh \n
        where dvolTPdh is the derivative of specific two-phase volume with enthalpy \n

        Return: volL
        '''
        volL = flowtools_obj.specificVolumeLiquid_Wrap(pressure, temperature, molar_weight,
                                                       liquid_molar_composition, density)

        return volL

    dx = 1e-5
    dvolLdxR = derivative(volL_func, x, dx, args=(p, T, MM))


    def hL_func(liquid_molar_composition, feed_molar_composition, pressure, temperature,
                pR, TR, hR, Cp, molar_weight, fluid_type='saturated_liquid'):
        '''
        This function has been created to make possible evaluate dvolTPdh \n
        where dvolTPdh is the derivative of specific two-phase volume with enthalpy \n

        Return: hL
        '''
        H_L_local = prop_obj.calculate_enthalpy(TR, temperature, pR, pressure, liquid_molar_composition,
                                          feed_molar_composition, hR, Cp, fluid_type)
        M_L_local = prop_obj.calculate_weight_molar_mixture(molar_weight, liquid_molar_composition, fluid_type)
        hL = H_L_local / M_L_local
        return hL

    dhLdxR = derivative(hL_func, x, dx, args=(z, p, T, pR, TR, hR, Cp, MM))

    volL = flowtools_obj.specificVolumeLiquid_Wrap(p, T, MM, x, twoPhase_models['density'])
    volG = flowtools_obj.specificVolumGas(p, T, MM, y)
    H_G = prop_obj.calculate_enthalpy(TR, T, pB_e, p, y, z, hR, Cp, 'saturated_vapor')
    M_V = prop_obj.calculate_weight_molar_mixture(MM, y, 'saturated_vapor')
    if y.all() == 0.0: hG = 0.0
    else: hG = H_G / M_V
    H_L = prop_obj.calculate_enthalpy(TR, T, pB_e, p, x, z, hR, Cp, 'saturated_liquid')
    M_L = prop_obj.calculate_weight_molar_mixture(MM, x, 'saturated_liquid')
    hL = H_L / M_L
    vol_fg = volG - volL
    h_fg = hG - hL
    dvolTPdh = (vol_fg - (1. - x[0]) * dvolLdxR) / (h_fg - (1. - x[0]) * dhLdxR)
    # ======================================================================
    #                      two-phase multiplier                            #
    # ======================================================================
    viscL_fo = flowtools_obj.viscosityLiquid_Wrap(p, T, z, z_mass, twoPhase_models['viscosity'])
    volL_fo = flowtools_obj.specificVolumeLiquid_Wrap(p, T, MM, z,twoPhase_models['density'])
    viscL = flowtools_obj.viscosityLiquid_Wrap(p, T, x, x_mass, twoPhase_models['viscosity'])
    volTP = flowtools_obj.specificVolumeTwoPhase(q, volG, volL)
    viscTP = flowtools_obj.viscosityTwoPhase(q, volG, volTP, viscL,twoPhase_models['viscosityTP'])
    phiLO2 = flowtools_obj.twoPhaseMultiplier(q, viscTP, viscL, volG, volL)
    logging.warning('densidade todo bifásico como liquido = %s' % (1. / volL_fo))
    logging.warning('densidade do bifásico = %s' % (1. / volTP))
    logging.warning('viscosidade todo bifásico como liquido = %s' % (viscL_fo * 1e6))
    logging.warning('viscosidade só da parte liquida = %s' % (viscL * 1e6))
    logging.warning('viscosidade do bifásico = %s' % (viscTP * 1e6))
    logging.warning('Multiplicador Bifasico = %s' % phiLO2)
    # ======================================================================
    #                      friction factor f_LO                            #
    # ======================================================================
    Re_mon = flowtools_obj.reynolds_function(Gt, Dc, viscL_fo)
    f_FLO = flowtools_obj.frictionFactorFanning_Wrap(Re_mon, ks, Dc, twoPhase_models['friction'])
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
    return np.array([dudl, dpdl, dhdl]) #array shape (3,)


# CREATING INITIAL CONDITIONS FOR THE TWO-PHASE BASED ON SINGLE-PHASE RESULTS
l_sp_0 = l_sp[-1]
u_sp_0 = u_sp[-1]
p_sp_0 = p_sp[-1]
h_sp_0 = h_sp[-1]
tolerance = np.array([1e-2, 1e-1, 1e-1])


# def main2():
#     return solve_ivp(edo_2p, [l_sp_0, L], [u_sp_0, p_sp_0, h_sp_0], method='Radau',  atol=tolerance, vectorized=False)


from scipy import integrate

# The ``driver`` that will integrate the ODE(s):
if __name__ == '__main__':

    # Start by specifying the integrator:
    # use ``vode`` with "backward differentiation formula"
    r = integrate.ode(edo_2p).set_integrator('vode', method='bdf')

    # Set the time range
    delta_l = 0.1
    # Number of time steps: 1 extra for initial condition
    num_steps = int(np.floor((L - l_sp_0) / delta_l)) + 1

    # Set initial condition(s): for integrating variable and time!
    r.set_initial_value([u_sp_0, p_sp_0, h_sp_0], l_sp_0)

    # Additional Python step: create vectors to store trajectories
    l_tp = np.zeros((num_steps, 1))
    u_tp = np.zeros((num_steps, 1))
    p_tp = np.zeros((num_steps, 1))
    h_tp = np.zeros((num_steps, 1))
    l_tp[0] = l_sp_0
    u_tp[0] = u_sp_0
    p_tp[0] = p_sp_0
    h_tp[0] = h_sp_0

    # Integrate the ODE(s) across each delta_t timestep
    k = 1
    while r.successful() and r.t <= L:
        r.integrate(r.t + delta_l)

        # Store the results to plot later
        l_tp[k] = r.t
        u_tp[k] = r.y[0]
        p_tp[k] = r.y[1]
        h_tp[k] = r.y[2]
        print('valor de l', l_tp[k])
        k += 1

    # All done!  Plot the trajectories in two separate plots:





# sys.exit(0)
# =====================================================================================
# ------------------------------EXECUTING main(two-phase)--------------------------   =
# uph_tp = main2()
# # =====================================================================================
# # UNPACKING THE RESULTS
# u_tp = uph_tp.y[0,:]
# p_tp = uph_tp.y[1,:]
# h_tp = uph_tp.y[2,:]
# l_tp = uph_tp.t
# PREPARING DATA TO PLOT
u_p = np.hstack((u_sp, u_tp))
p_p = np.hstack((p_sp, p_tp))
h_p = np.hstack((h_sp, h_tp))
l_p = np.hstack((l_sp, l_tp))
#
# STACKING SP results to TP ones and taking the transpose
uph_sp_plus_tp = (np.hstack((uph_sp.y, r.y))).T
l_sp_plus_tp = (np.hstack((uph_sp.t, r.t))).T
l_size = l_sp_plus_tp.shape[0]
# Forcing vector length to become two dimensional
l_sp_plus_tp.shape = (l_size, 1)
# Appending l_sp_plus_tp to uph_sp_plus_tp
tuph_sp_plus_tp = np.append(l_sp_plus_tp, uph_sp_plus_tp, axis=1)
# Send matrix to pandas
df_sp_plus_tp = pd.DataFrame(tuph_sp_plus_tp, columns=['length(m)', 'velocity(m/s)', 'pressure(Pa)', 'enthalpy(J/kgK)'])
entire_case = 'sp_AND_tp'
# r na frente é devido a problematica da barra invertida do windows
df_sp_plus_tp.to_csv(r'..\Results_pkg\{}.csv'.format(entire_case), sep='\t', float_format='%.4f')


'''
==========================================================
=                      PLOTTING                          =
==========================================================
'''
# CpL = flowtools_obj.specificLiquidHeat_jpDias(T_e, p_e, z_mass)
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
# volL_inter = flowtools_obj.specificVolumeLiquid_Wrap(pinter, Tinter, MM, z, z_mass, density_models['_jpDias'])
# densL_inter = 1. / volL_inter
# print('olha a densidade do ELV', densL_inter)
# print('veja como é meu z = ', z, 'e também z_mass', z_mass)
# Acinter = Area(0.640)
# Gtinter = mdotL_e / Acinter
# rinho = np.linspace(-D / 2, D / 2, 101)
#
# # q = 0.4
# # viscF = flowtools_obj.viscosityLiquid_Wrap(p, T, z, z_mass, 'jpDias')
# # volG = flowtools_obj.specificVolumGas(p, T, MM, z)
# # volF = flowtools_obj.specificVolumeLiquid_Wrap(p, T, MM, z, z_mass, density_model='jpDias')
# # vol2P = flowtools_obj.specificVolumeTwoPhase(q, volG, volF)
# # visc2P = flowtools_obj.viscosityTwoPhase(q,volG, vol2P, viscF, viscTP_model='Dukler')
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
# # plt.plot(l, pB_e_v)
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
