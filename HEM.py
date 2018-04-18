# -*- coding: utf-8 -*-
import logging
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
# D: diametro em metros [m]
# Ld: comprimento total do duto [m]
# pointsNumber: quantidade divisoes do duto, ou seja, quantidade de Z steps [-]
# ks: absolut rugosity [m]
# rug = ks/D: rugosidade relativa [-]
# angleVenturi_in: entrance venturi angle [rad]
# angleVenturi_out: outlet venturi angle [rad]  
# Dvt: venturi throat diameter [m] 
# ziv: coordinate where venturi begins [m] 
# zig: coordinate where venturi throat begins [m] 
# zfg: coordinate where venturi throat ends [m] 
# zfv: coordinate where venturi ends [m] 
# Ac: cross area section [m2] -- valid for any duct position, including z position inside venturi, i.e., Ac = Ac(z) 
# Ac_pipe: pipe cross area section [m2] -- valid just for pipe area; so, can be calculated at pipe entrance, z = 0
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

LC, base = 0.8/100, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
xRe, xRe_mass = Tools_Convert.frac_input(MM, zin, base)
hR = hR_mass * prop_obj.calculate_weight_molar_mixture(MM, xRe, 'saturated_liquid')
sR = sR_mass * prop_obj.calculate_weight_molar_mixture(MM, xRe, 'saturated_liquid')
# pG = 1.2 * bubble_obj.pressure_guess(T_e, xRe)
# pB, y, Sy, counter = bubble_obj(T_e, xRe)
# y_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, y)
MMixture = prop_obj.calculate_weight_molar_mixture(MM, xRe,"liquid")

'''
=================================================================================================================
CREATING MORE NECESSARY OBJECTS
=================================================================================================================
'''
eos_obj = PengRobinsonEos(pC, Tc, AcF, omega_a, omega_b, kij)
hsFv_obj = HSFv(pC, TR, Tc, AcF, Cp, MM, hR, sR) #to obtain enthalpy


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



#[2]========== OUTRAS CONSTANTES + CÁLCULO SIMPLES ===========    
Ac_e = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, 0.0)
Gt_e = mdotL_e / Ac_e
densL_e = prop_obj.calculate_density_phase(p_e, T_e, MM, xRe, "liquid") 
spvolL_e = np.power(densL_e,-1) 
                           


'''
=================================================================================================================
CREATING ANOTHER NECESSARY OBJECT
=================================================================================================================
'''

FlowTools_obj = FlowTools_class(D, Gt_e)
viscL_e = FlowTools_obj.viscosidadeMonofasico(T_e, p_e, xRe)
print('viscosidade líquido subresfriado - entrada duto [1e-6 Pa.s] ', viscL_e * 1e6)
#print(twoPhaseFlowTools_obj)


'''
=================================================================================================================
FRICTION FACTOR - CHURCHILL (1977)
=================================================================================================================
'''
def fanningFactor(Re_mon, ks, diametro):
    '''
    Churchill equation to estimate friction factor (pg 149, Ron Darby's book) \n
    Can be applied for all flow regimes in single phase \n
    Re_mon: Single phase Reynolds number [-] \n
    ks: rugosity [m] \n
    diametro: diameter [m]
     '''
    f1 = 7. / Re_mon
    f2 = 0.27 * ks/diametro
    a = 2.457 * np.log(1. / (f1**0.9 + f2))
    A = a ** 16
    b = 37530. / Re_mon
    B = b ** 16
    f3 = 8. / Re_mon
    return  2 * (f3 ** 12 + 1./(A + B) ** 1.5 ) ** (1. / 12)



#[6]============================ MAIN - ODE's system =========================================
#source: The EDO's system wrote here was based on page 167  Ghiaasiaan
# How to solve this system? See the page -->
# --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

geometric_param = (angleVenturi_in, angleVenturi_out, ks, D, Dvt, ziv, zig, zfg, zfv)

def systemEDOsinglePhase(uph, Zduct, mdotL_e, MMixture, h_e, T_e, Cp, xRe, deltaZ, geometric_param):
    '''
    [1] - Objetivo: resolver um sistema de EDO's das derivadas velocidade, presssão e entalpia com o uso da ferramenta linalg.solve; \t
    [2] - Input: \t
        [2.a] vetor comprimento duto (Zduct) \t
        [2.b] fluxo mássico total baseado na área entrada (Gt) \t
        [2.c] diâmetro (D) \t
        [2.d] peso molecular da mixtura (MMixture) \t
        [2.e] viscosidade da líquido (viscL) \t
        [2.f] rugosidade absoluta (ks) \t
        [2.g] entalpia entrada duto (h_e) \t
        [2.h] temperatura entrada duto (T_e) \t
        [2.i] vetor calor específico (Cp = ([CpR, CpO])) \t
        [2.j] vetor concentração molar entrada duto (  xRe = ([xR, xO])  ) \t
        [2.l] incremento/passo no comprimento Z do duto (deltaZ); saída adicional do np.linspace \t
        [2.m] uma lista com os parametros geométricos do circuito escoamento (geometric_param)
    [3] - Output: du/dz, dp/dz e dh/dz
    '''
 
    u, p, h = uph
    CpL = np.einsum('i,i', xRe, Cp)  
    T = T_e + (h - h_e) / CpL
    (angleVenturi_in, angleVenturi_out, ks, D, Dvt, ziv, zig, zfg, zfv) = geometric_param
    Ac = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, Zduct)
    Dc = np.sqrt(4 * Ac / np.pi)
    Gt = mdotL_e / Ac
    Gt2 = np.power(Gt, 2)
    viscL = FlowTools_obj.viscosidadeMonofasico(T, p, xRe)
    
    densL_e = prop_obj.calculate_density_phase(p_e, T_e, MM, xRe, "liquid")
    spvolL_e = np.power(densL_e, -1) 
    densL = prop_obj.calculate_density_phase(p, T, MM, xRe, "liquid")
    spvolL = np.power(densL, -1) 
    
    #derivative approximation
    if (T - T_e) != 0.0: spvolLdT = (spvolL - spvolL_e) / (T - T_e) 
    else: spvolLdT = 0.0 #avoid division by zero

    avrg_spvolL = (spvolL + spvolL_e) / 2.
    beta = np.power(avrg_spvolL, -1) * spvolLdT #appears at mass conservation (eq. 3.33) and EDO's matrix (eq. 3.40)
    print('Z = ', Zduct, '(T - T_e)', (T - T_e), file=fh)
    
    A1st = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, Zduct)
    A2nd = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, Zduct + deltaZ)
    deltaA = A2nd - A1st
    avrgA = (A2nd + A1st) / 2.
    dAdZ = deltaA / deltaZ
    # print('Z = ', Zduct, '\t\t Dc = ', Dc, '\t\t A1st', A1st, 'A2nd', A2nd, '\t\t dAdZ', dAdZ, file=fh)



    A11, A12, A13 = np.power(u,-1), 0., (- beta / CpL)     
    A21, A22, A23 = u, spvolL, 0.
    A31, A32, A33 = u, 0., 1.

    Re_mon = Gt * Dc / viscL   
    # print('Z = ', Zduct, '\t Gt = ', Gt, file=fh)
    # print('Z = ', Zduct, '\t Re = ', Re_mon, file=fh)
    colebrook = lambda f0 : 1.14 - 2. * np.log10(ks / Dc + 9.35 / (Re_mon * np.sqrt(f0))) -1 / np.sqrt(f0) 
    f_D = optimize.newton(colebrook, 0.02)  #Darcy 
    if Re_mon > 3000.: f_F = f_D / 4.
    else: f_F = 16. / Re_mon

    # f_F = fanningFactor(Re_mon, ks, Dc)

    aux = -2.0 * Gt2 * spvolL * (f_F / Dc )
    C1, C2, C3 = (- dAdZ / avrgA), aux, aux
    # print('Z =', Zduct, '\t C1 = ', C1, file=fh)
    
    
    matrizA = np.array([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])
    RHS_C = np.array([C1, C2, C3])
    dudz, dpdz, dhdz = np.linalg.solve(matrizA, RHS_C)
    return [dudz, dpdz, dhdz]

 

#[7] ==============FUNÇÃO A SER INTEGRADA =================
#source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

#=================== CREATING VECTORS POINTS =============================
pointsNumber = 1000
Zduct, deltaZ = np.linspace(0, Ld, pointsNumber + 1, retstep=True)


# for j in Zduct:
#     if j == Ld:
#         break
#     Ac1 = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, j)
#     Ac2 = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, j + deltaZ)
#     deltaAc = Ac2 - Ac1
#     #if deltaAc != 0.0:
#     print('valor de z = ', j + deltaZ, 'variação de área', deltaAc)


#==================== BUILDING INITIAL VALUES ============
u_e = (mdotL_e * spvolL_e) / Ac_e                  # Para obter u_e (subcooled liquid)
F_V, h_e, s_e = hsFv_obj(p_e, T_e, xRe)       #Para obter h_e (subcooled liquid, so F_V = 0)
uph_0 = [u_e, p_e, h_e]



#===================== INTEGRATION ========================
fh = open("saida_.txt","w")
zcritico = np.array([ziv, zig, zfg, zfv]) 
uph_singlephase = integrate.odeint(systemEDOsinglePhase, uph_0, Zduct, args=(mdotL_e, MMixture, h_e, T_e, Cp, xRe, deltaZ, geometric_param), tcrit = zcritico)
fh.close()


# #[8] ================= TAKING THE RESULTS =================
u = uph_singlephase[:,0]
p = uph_singlephase[:,1]
h = uph_singlephase[:,2]
# pB_v = pB * np.ones_like(Zduct)
#alfa = FlowTools_obj.fracaoVazio(0.01, p,T_e,MMixture, spvolL_e)


CpL = np.einsum('i,i', xRe, Cp)  # -- capacidade térmica líquido subresfriado [J/(kg K)] (@ solução ideal)
T = T_e + (h - h_e) / CpL











# #[9]=========================== PLOT =====================


# plt.figure(figsize=(7,5))
# #plt.ylim(20,120)
# plt.xlabel('Z [m]')
# plt.ylabel('T [K]')
# plt.plot(Zduct, T)
# plt.legend(['Temperatura Esc. Incompressivel'], loc=3)


plt.figure(figsize=(7,5))
#plt.xlim(0.6,0.8)
plt.xlabel('Z [m]')
plt.ylabel('u [m/s]')
plt.plot(Zduct, u)
plt.legend(['$u_{incompressivel}$ ao longo do duto'], loc=1) #loc=2 vai para canto sup esq


plt.figure(figsize=(7,5))
#plt.xlim(0.675,0.8)
# plt.ylim(8e5, 10e5)
plt.xlabel('Z [m]')
plt.ylabel('P [Pascal]')
plt.plot(Zduct, p)
plt.legend(['Pressao Esc. Incompressivel'], loc=3)
# plt.plot(Zduct, pB_v)
# plt.legend(['Pressao Esc. Incompressivel', 'Pressão Saturação'], loc=3)

# plt.figure(figsize=(7,5))
# #plt.ylim(20,120)
# plt.xlabel('Z [m]')
# plt.ylabel('H [J/kg]')
# plt.plot(Zduct, h)
# plt.legend(['$h_{incompressivel}$ ao longo do duto'], loc=3)


plt.show()
plt.close('all')
