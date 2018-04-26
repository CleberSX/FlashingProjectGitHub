# -*- coding: utf-8 -*-
import logging, sys
import numpy as np 
from scipy import integrate, optimize
from scipy.misc import derivative
import matplotlib.pyplot as plt
import Tools_Convert
from InputData___ReadThisFile import props
from References_Values import input_reference_values_function
from BubbleP import bubble_obj
from Properties import Properties
from EOS_PengRobinson import PengRobinsonEos
from EnthalpyEntropy import HSFv
from input_flow_data import input_flow_data_function as FlowData
from input_pipe_data import input_pipe_data_function as PipeData
from input_pipe_data import areaVenturiPipe_function as Area
from FlowTools_file import FlowTools_class


#1st OPTION:
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
#2nd OPTION
#logging.basicConfig(filename='Cleber_File.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.disable(logging.CRITICAL)



#Constantes
#g = 9.8                             # m/s^2
R = 8314.34                         # J / (kmol K)



"""
#======================== PARAMETROS DE ENTRADA (O PROGRAMA TODO no SI) ======================
"""
#[1]================= DADOS DE ENTRADA ================= 
#
#mdotG: taxa (tx) massica de gás saturado [kg/s] {@ saturation}
#mdotF: tx massica de líquido saturado [kg/s] {@ saturation}
#mdotL_e: tx massica de líquido subresfriado entrada duto [kg/s] {@ subcooled liquid} 
#MMixture: molecular weight - peso molecular da mistura (entrada duto) [kg/kmol]
#T_e: temperatura de entrada [K]
#p_e: pressao de entrada [Pa]
#pB: pressão de saturação [Pa] @ pB = pB(T_e)
#pB_v: é o próprio pB, porém na forma vetorial 
#x_e: título de vapor entrada duto [-]
#mdotT = mdotG + mdotF: tx massica total (kg/s)
#densL, densF, densG: subcooled liquid density [kg/m3], satureted liquid density [kg/m3], satureted gas density [kg/m3]
#spvolL, spvolF, spvolG: specific volume subcooled liquid [m3/kg], specific volume satureted liquid [m3/kg], specific volume satureted gas [m3/kg]
#densL_e, spvolL_e: subcooled liquid density at duct entrance [kg/m3], specific volume subcooled liquid at duct entrance [m3/kg] 
#spvolTP: specific volume two-phase [m3/kg]
#viscL: viscosidade do líquido subresfriado [kg/(m.s)] 
#viscF: viscosidade do líquido saturado [kg/(m.s)] {@ saturation}
#visG: viscosidade do gás saturado [kg/(m.s)] {@ saturation}
#Gt: fluxo massico superficial total ((kg/s)/m2)
#u: flow speed [m/s]
#uG: satureted gas speed [m/s]
#uF: satureted liquid speed [m/s]
#Sr = uG/uF: speed ratio
#LC: light component (our refrigerant)
#Z: compressibility factor --> [p*spvolG = (Z*R*T) / MM]
#xRe: vector binary mixture molar composition at pipe's entrance: xRe = ([refrigerant]_e,[oil]_e) 
#xR: vector binary mixture molar composition at some pipe's positon: xR = ([refrigerant],[oil]) 
#Zduct: duct length - [m]
#f_D: Darcy friction factor
#f_F: Fanning friction factor
# hR_mass: reference enthalpy in mass base [J/kg]
# hR: reference enthalpy (in molar base) [J/kmol]
# sR_mass: reference entropy in mass base [J/kg K]
# sR: reference entropy (in molar base) [J/kmol K]
# CpL: subcooled liquid's specific heat [J/(kg K)] -- CpL = np.eisum('i,i', xRe, Cp)
# Cp: vector components' specific heat, which Cp1= Cp[0] and Cp2 = Cp[1]
# D: diameter for duct just (it isn't represent Venturi); it is a constant -- see Dvt  [m]
# Ld: total lenght [m]
# pointsNumber: quantidade divisoes do duto, ou seja, quantidade de Z steps [-]
# ks: absolut rugosity [m]
# rug = ks/D: rugosidade relativa [-]
# angleVenturi_in: entrance venturi angle [rad]
# angleVenturi_out: outlet venturi angle [rad]  
# Dvt: venturi throat diameter where Dvt change according position, i.e, Dvt = Dvt(z) [m] 
# ziv: coordinate where venturi begins [m] 
# zig: coordinate where venturi throat begins [m] 
# zfg: coordinate where venturi throat ends [m] 
# zfv: coordinate where venturi ends [m] 
# Ac: cross section area; applied for entire circuit flow, i.e, Ac = Ac(z) [m2]
# rc: cross section radius [m]; applied for entire circuit flow, i.e, rc = rc(z) [m]
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
(pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props
(TR, hR_mass, sR_mass) = input_reference_values_function()
(p_e, T_e, mdotL_e) = FlowData() # <=============================== change here
(angleVenturi_in, angleVenturi_out, ks, Ld, D, Dvt, ziv, zig, zfg, zfv) = PipeData() # <=============================== change here



'''
=================================================================================================================
CREATING NECESSARY OBJECT
=================================================================================================================
'''
prop_obj = Properties(pC, Tc, AcF, omega_a, omega_b, kij)
#print(prop_obj)

'''
=================================================================================
TAKING SATURATION PRESSURE FROM BubbleP.py
FOR MORE INFORMATION ABOUT Bubble Pressure consult BubbleP.py
=================================================================================
'''

def solution_concentration_set_function(lightComponent, MM, base):
    '''
    lightComponent: the subcooled liquid solution concentration is set up ...
                ... based on lighter component concentration \t
    lightComponent: example 5./100, 0.05/100, 30./100
    base: you must write 'mass' or 'molar'
    '''
    zin = np.array([lightComponent, (1. - lightComponent)])
    xRe, xRe_mass = Tools_Convert.frac_input(MM, zin, base)
    return xRe, xRe_mass

xRe, xRe_mass = solution_concentration_set_function((0.05/100), MM, 'mass')


def saturationPressure_ResidualProperties_MolarMixture():
    '''callling some thermodynamics variables
    All of them are objects 
    '''
    hR = hR_mass * prop_obj.calculate_weight_molar_mixture(MM, xRe, 'saturated_liquid')
    sR = sR_mass * prop_obj.calculate_weight_molar_mixture(MM, xRe, 'saturated_liquid')
    # pG = 1.2 * bubble_obj.pressure_guess(T_e, xRe)
    pB, _y, _Sy, _counter = bubble_obj(T_e, xRe)
    #pB = bubble_obj(T_e, xRe)[0]
    # y_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, y)
    MMixture = prop_obj.calculate_weight_molar_mixture(MM, xRe,"liquid")
    return (hR, sR, pB, MMixture)

(hR, sR, pB, MMixture) = saturationPressure_ResidualProperties_MolarMixture()



'''
=================================================================================================================
CREATING MORE NECESSARY OBJECTS
=================================================================================================================
'''
eos_obj = PengRobinsonEos(pC, Tc, AcF, omega_a, omega_b, kij)
hsFv_obj = HSFv(pC, TR, Tc, AcF, Cp, MM, hR, sR) #to obtain enthalpy

FlowTools_obj = FlowTools_class(mdotL_e)
# viscL_e = FlowTools_obj.viscosidadeMonofasico(T_e, p_e, xRe)
# viscL_jpDias = FlowTools_obj.jpDias_liquidViscosity(T_e, p_e, xRe_mass)

# densL_jpDias = FlowTools_obj.jpDias_liquidPhaseDensity(T_e, p_e, xRe_mass)
# densL_e_ELV = prop_obj.calculate_density_phase(p_e, T_e, MM, xRe, "liquid")

# print('viscosidade líquido subresfriado - entrada duto Dalton [1e-6 Pa.s] ', viscL_e * 1e6)
# print('viscosidade líquido subresfriado - entrada duto jpDias [1e-6 Pa.s] ', viscL_jpDias * 1e6)
# print('densidade líquido subresfriado - entrada duto jpDias [kg m-3] ', densL_jpDias)
# print('densidade líquido subresfriado - entrada duto ELV [kg m-3] ', densL_e_ELV)
#print(twoPhaseFlowTools_obj)


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
def initialValues_function():
    '''necessary in the Odeint numerical method
    Return: u_e, p_e, h_e
    '''
    Ac_e, _rc_e = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, 0.0)
    densL_e = prop_obj.calculate_density_phase(p_e, T_e, MM, xRe, "liquid") 
    spvolL_e = np.power(densL_e,-1) 
    u_e = (mdotL_e * spvolL_e) / Ac_e                  # Para obter u_e (subcooled liquid)
    _F_V, h_e, _s_e = hsFv_obj(p_e, T_e, xRe)            #Para obter h_e (subcooled liquid, so F_V = 0)
    return (u_e, p_e, h_e)

(u_e, p_e, h_e) = initialValues_function()
uph_0 = [u_e, p_e, h_e]



#[6]============================ MAIN - ODE's system =========================================
#source: The EDO's system wrote here was based on page 167  Ghiaasiaan
# How to solve this system? See the page -->
# --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

geometric_list = (angleVenturi_in, angleVenturi_out, ks, D, Dvt, ziv, zig, zfg, zfv)
flow_list = (mdotL_e, MM, h_e, T_e, Cp, xRe, xRe_mass)
def systemEDOsinglePhase(uph, Zduct, deltaZ, flow_list, geometric_list):
    '''
    Objetivo: resolver um sistema de EDO's em z para u, p e h com linalg.solve; \t
        [1] - Input: \t
            [1.a] vetor comprimento circuito (Zduct) \t
            [1.b] vazão mássica líquido subresfriado entrada (mdotL_e) \t
            [1.c] diâmetro tubo (D) \t
            [1.d] peso molecular de cada componente (MM) \t
            [1.e] viscosidade da líquido (viscL) \t
            [1.f] rugosidade absoluta (ks) \t
            [1.g] entalpia entrada duto (h_e) \t
            [1.h] temperatura entrada duto (T_e) \t
            [1.i] vetor calor específico (Cp = ([CpR, CpO])) \t
            [1.j] vetor concentração molar entrada duto (  xRe = ([xR, xO])  ) \t
            [1.l] vetor concentração massica entrada duto (  xRe_mass = Tools.convert...  ) \t
            [1.m] incremento/passo no comprimento Z do duto (deltaZ); saída adicional do np.linspace \t
            [1.n] uma lista com parâmetros do escoamento (flow_list) \t
            [1.o] uma lista com os parametros geométricos do circuito escoamento (geometric_list)
        [2] - Return: [du/dz, dp/dz, dh/dz]
    '''
    
    #unpack
    u, p, h = uph
    (mdotL_e, MM, h_e, T_e, Cp, xRe, xRe_mass) = flow_list
    (angleVenturi_in, angleVenturi_out, ks, D, Dvt, ziv, zig, zfg, zfv) = geometric_list
    
    #simple computations
    CpL = FlowTools_obj.jpDias_liquidSpecificHeat(T_e, p_e, xRe_mass)
    T = T_e + (h - h_e) / CpL
    Ac, rc = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, Zduct)
    Dc = 2. * rc
    Gt = mdotL_e / Ac
    Gt2 = np.power(Gt, 2)
    
    
    
    def therm_function(density_model='ELV'):
        '''this function calls external thermal (objects) tools \n
        
        Return: spvolL_e, spvolL'''
        density_models = ['ELV', 'jpDias']
        if density_model not in density_models:
            msg = 'Invalid density model in --> %s' % sys._getframe().f_code.co_name
            msg += '\t Choose one of the models: %s' % density_models
            raise Exception(msg)
        if density_model == 'ELV':
            densL_e = prop_obj.calculate_density_phase(p_e, T_e, MM, xRe, "liquid")
            densL = prop_obj.calculate_density_phase(p, T, MM, xRe, "liquid")
        elif density_model == 'jpDias':
            densL_e = FlowTools_obj.jpDias_liquidPhaseDensity(T_e, p_e, xRe_mass)
            densL = FlowTools_obj.jpDias_liquidPhaseDensity(T, p, xRe_mass)
        spvolL_e = np.power(densL_e, -1) 
        spvolL = np.power(densL, -1) 
        return spvolL_e, spvolL

    spvolL_e, spvolL = therm_function(density_model='ELV')

    def beta_function():
        '''this function calculates thermal expansion coefficient - beta \n
        
        Return: beta value
        '''
        if (T - T_e) != 0.0: spvolLdT = (spvolL - spvolL_e) / (T - T_e) 
        else: spvolLdT = 0.0 
        avrg_spvolL = (spvolL + spvolL_e) / 2.
        return np.power(avrg_spvolL, -1) * spvolLdT
    
    beta = beta_function()

    def dAdZ_function():
        '''derivative approx. to evaluate dAdZ [m2/m] and area average [m2] \n
        dAdZ: delta Area / delta Z \n 
        avrgA: average Area \n 

        Return: dAdZ, avrgA
        '''
        A1st, _rc1st = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, Zduct)
        A2nd, _rc2nd = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, Zduct + deltaZ)
        deltaA = A2nd - A1st
        avrgA = (A2nd + A1st) / 2.
        return (deltaA / deltaZ), avrgA

    dAdZ, avrgA = dAdZ_function()

    
    Re_mon = FlowTools_obj.reynolds_function(Gt, Dc, p, T, xRe, xRe_mass, visc_model='NISSAN')
    f_F = FlowTools_obj.frictionFactorFanning(Re_mon, ks, Dc, friction_model='Colebrook')
    if np.abs(Zduct - 0.640) < 1e-2:
        print('Em Z = ', Zduct, 'Re = ', Re_mon, 'Veloc = ', u) 
    
    A11, A12, A13 = np.power(u,-1), 0., (- beta / CpL)     
    A21, A22, A23 = u, spvolL, 0.
    A31, A32, A33 = u, 0., 1.


    aux = -2.0 * Gt2 * np.power(spvolL, 2) * (f_F / Dc )
    C1, C2, C3 = (- dAdZ / avrgA), aux, aux
    
    
    matrizA = np.array([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])
    RHS_C = np.array([C1, C2, C3])
    dudz, dpdz, dhdz = np.linalg.solve(matrizA, RHS_C)
    return [dudz, dpdz, dhdz]

 

#[7] ==============FUNÇÃO A SER INTEGRADA =================
#source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

#=================== CREATING VECTORS POINTS =============================
pointsNumber = 1000
Zduct, deltaZ = np.linspace(0, Ld, pointsNumber + 1, retstep=True)



#===================== SOLVING - INTEGRATING ========================
# fh = open("saida_.txt","w")
Zduct_crit = np.array([ziv, zig, zfg, zfv]) 
uph_singlephase = integrate.odeint(systemEDOsinglePhase, uph_0, Zduct, 
                    args=(deltaZ, flow_list, geometric_list), tcrit = Zduct_crit)
# fh.close()


# #[8] ================= UNPACKING THE RESULTS =================
u = uph_singlephase[:,0]
p = uph_singlephase[:,1]
h = uph_singlephase[:,2]
pB_v = pB * np.ones_like(Zduct)


def taking_SpecificGeometricPosition(array, target):
    '''
    This function will find an interested value in a vector \n 

    target: a specific position [same vector's unit] \n
    
    E.g., in case vector position Zduct = 150mm, you must use: target = 0.150

    Return: index  (the index/position of the interested value in the vector)
     '''
    index = (np.abs(array - target)).argmin()
    return index






CpL = FlowTools_obj.jpDias_liquidSpecificHeat(T_e, p_e, xRe_mass)
T = T_e + (h - h_e) / CpL










# #[9]=========================== PLOTTING =====================
 
index_Z_150mm = taking_SpecificGeometricPosition(Zduct, 0.150)
# print('taking index of vector where Zduct = 150mm', index_Z_150mm) 

index_Z_640mm = taking_SpecificGeometricPosition(Zduct, 0.640)
print('taking index of vector where Zduct = 640mm', index_Z_640mm) 
pinter = p[index_Z_640mm]
hinter = h[index_Z_640mm]
Tinter = T_e + (hinter - h_e) / CpL
densL_inter = prop_obj.calculate_density_phase(pinter, Tinter, MM, xRe, "liquid") 
spcfVolL_inter = np.power(densL_inter, -1)
Acinter, rcinter = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, 0.640)
Gtinter = mdotL_e / Acinter
rinho = np.linspace(-D/2, D/2, 101)

R = D/2.
ur = np.zeros_like(rinho)
for i,r in enumerate(rinho):
    ur[i] = 2 * Gtinter * spcfVolL_inter * (1. - (r/R)**2)
    

# print(ur)
# ucopy = np.copy(ur)
# utotal = np.concatenate((ur, ucopy))
# rinhocopy = - np.copy(rinho)
# rtotal = np.concatenate((rinho, rinhocopy))

plt.figure(figsize=(7,5))
#plt.xlim(0.6,0.8)
plt.ylabel('r [mm]')
plt.xlabel('u [m/s]')
plt.plot(ur, rinho * 1e3)
# plt.legend(['$u_{incompressivel}$ ao longo do duto'], loc=1)


# plt.figure(figsize=(7,5))
# #plt.ylim(20,120)
# plt.xlabel('Z [m]')
# plt.ylabel('T [K]')
# plt.plot(Zduct, T)
# plt.legend(['Temperatura Esc. Incompressivel'], loc=3)


# plt.figure(figsize=(7,5))
# #plt.xlim(0.6,0.8)
# plt.xlabel('Z [m]')
# plt.ylabel('u [m/s]')
# plt.plot(Zduct, u)
# plt.legend(['$u_{incompressivel}$ ao longo do duto'], loc=1) #loc=2 vai para canto sup esq



# plt.figure(figsize=(7,5))
# plt.title('Grafico comparativo com pg 81 Tese Dalton Gt baseado 16mm')
# plt.grid(True)
# plt.xlabel('Z* [m]')
# plt.ylabel('p - p#1 [Pascal]')
# # plt.plot((Zduct[(point_Zduct150mm-1):] - 0.150), (p[(point_Zduct150mm - 1):] - p150mm))
# plt.plot((Zduct[index_Z_150mm: ] - 0.150), (p[index_Z_150mm: ] - p[index_Z_150mm]))
# plt.xlim(0.0, 0.9)
# # plt.ylim(8e5, 10e5)
# # plt.legend(['Pressao Esc. Incompressivel'], loc=3)
# # plt.plot(Zduct, pB_v)
# # plt.legend(['Pressao Esc. Incompressivel', 'Pressão Saturação'], loc=3)
# plt.legend('Pressão Esc. Incompressível', loc=1)

# plt.figure(figsize=(7,5))
# #plt.ylim(20,120)
# plt.xlabel('Z [m]')
# plt.ylabel('H [J/kg]')
# plt.plot(Zduct, h)
# plt.legend(['$h_{incompressivel}$ ao longo do duto'], loc=3)


plt.show()
plt.close('all')
