# -*- coding: utf-8 -*-
import numpy as np 
from scipy import integrate, optimize
import matplotlib.pyplot as plt
import Tools_Convert
from InputData___ReadThisFile import props
from References_Values import input_reference_values_function
from BubbleP import bubble_obj
from Properties import Properties
from EOS_PengRobinson import PengRobinsonEos
from EnthalpyEntropy import HSFv
from input_flow_data import input_flow_data_function
from input_pipe_data import input_pipe_data_function
from TwoPhaseFlowTools_file import twoPhaseFlowTools_class

#Constantes
#g = 9.8                             # m/s^2
R = 8314.34                         # J / (kmol K)
Sr = 1.


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
#D: diametro em inch transformado para metros [m]
#Ld: comprimento total do duto [m]
#pointsNumber + 1: quantidade divisoes do duto, ou seja, quantidade de Z steps [-]
#rks: ugosidade absoluta [m]
#teta: ângulo inclinação medido a partir da horizontal(+ para escoam. ascend.) [graus]
#densL, densF, densG: subcooled liquid density [kg/m3], satureted liquid density [kg/m3], satureted gas density [kg/m3]
#spvolL, spvolF, spvolG: specific volume subcooled liquid [m3/kg], specific volume satureted liquid [m3/kg], specific volume satureted gas [m3/kg]
#densL_e, spvolL_e: subcooled liquid density at duct entrance [kg/m3], specific volume subcooled liquid at duct entrance [m3/kg] 
#spvolTP: specific volume two-phase [m3/kg]
#viscF: viscosidade do líquido saturado [k/(m.s)] {@ saturation}
#visG: viscosidade do gás saturado [k/(m.s)] {@ saturation}
#Gt: fluxo massico superficial total ((kg/s)/m2)
#rug = ks/D: rugosidade relativa [-]
#A: área transversal duto [m2]
#u: flow speed [m/s]
#uG: satureted gas speed [m/s]
#uF: satureted liquid speed [m/s]
#Sr = uG/uF: speed ratio
#LC: light component (our refrigerant)
#Z: compressibility factor --> [p*spvolG = (Z*R*T) / MM]
#xRe: binary mixture composition at entrance: xRe = ([refrigerant]_e,[oil]_e) where [brackets]_e means concentration at entrance - [kg Refrig / kg mixture]_e
#xR: binary mixture composition at some positon: xR = ([refrigerant],[oil]) 
#Zduct: duct length - [m]
#f_D: Darcy friction factor
#f_F: Fanning friction factor
# hR_mass: reference enthalpy in mass base [J/kg]
# hR: reference enthalpy (in molar base) [J/kmol]
# sR_mass: reference entropy in mass base [J/kg K]
# sR: reference entropy (in molar base) [J/kmol K]


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
(p_e, T_e, mdotL_e, viscG, viscF) = input_flow_data_function() # <=============================== change here
(D, Ld, ks) = input_pipe_data_function() # <=============================== change here


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

LC, base = 99./100, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
xRe, xRe_mass = Tools_Convert.frac_input(MM, zin, base)
hR = hR_mass * prop_obj.calculate_weight_molar_mixture(MM, xRe, 'saturated_liquid')
sR = sR_mass * prop_obj.calculate_weight_molar_mixture(MM, xRe, 'saturated_liquid')
pG = 1.2 * bubble_obj.pressure_guess(T_e, xRe)
pB, y, Sy, counter = bubble_obj(T_e, xRe)
y_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, y)
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

if __name__== '__main__':
    print('\n---------------------------------------------------')
    print('[1] - Guess pB [Pa]= %.8e' % pG)
    print('[2] - ======> at T = %.2f [C], pB = %.8e [Pa] ' % ((T_e - 273.15), pB) + '\n')
    print('[3] - Concentration vapor phase [molar] = ', y.round(3))
    print('[4] - Concentration vapor phase [mass] = ', y_mass.round(3))
    print('[5] - Pay attention if Sy is close to unity (Sy = %.10f) [molar]' % Sy)
    print('[6] - Feed global {mass} fraction = ', xRe_mass.round(3))
    print('[7] - Feed global {molar} fraction = ', xRe.round(3))
    print()



#[2]========== OUTRAS CONSTANTES + CÁLCULO SIMPLES ===========
A = np.pi * np.power(D, 2) / 4      
rug = ks / D
Gt = mdotL_e / A  
densL_e = prop_obj.calculate_density_phase(p_e, T_e, MM, xRe, "liquid") 
spvolL_e = np.power(densL_e,-1) 
viscL = viscF        #<--- considerando que a visc do líquido subresfriado é igual ao do líquido saturado                                   
                           


'''
=================================================================================================================
CREATING ANOTHER NECESSARY OBJECT
=================================================================================================================
'''

twoPhaseFlowTools_obj = twoPhaseFlowTools_class(MM, viscG, viscF, D, Gt)

print(twoPhaseFlowTools_obj)




#[5] == PROPRIEDADES CALOR & GÁS 
CpL = np.einsum('i,i', xRe, Cp)     # -- capacidade térmica líquido subresfriado [J/(kg K)] (@ solução ideal)





#[6]============================ MAIN - ODE's system =========================================
#source: The EDO's system wrote here was based on page 167  Ghiaasiaan
# How to solve this system? See the page -->
# --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
def systemEDOsinglePhase(uph, Zduct, Gt, D, MMixture, viscL, ks, h_e, T_e, CpL):
    u, p, h = uph
    T = T_e + (h - h_e) / CpL
    print('temperatura T = \t',T, 'temperatura T_e = \t', T_e)
    Gt2 = np.power(Gt, 2)
    Re_mon = Gt * D / viscL   
    densL = prop_obj.calculate_density_phase(p, T, MM, xRe, "liquid")
    spvolL = np.power(densL, -1) 
    

    A11, A12, A13 = np.power(u,-1), 0., 1.e-5    
    A21, A22, A23 = u, spvolL, 0.
    A31, A32, A33 = u, 0., 1.
        
    colebrook = lambda f0 : 1.14 - 2. * np.log10(ks / D + 9.35 / (Re_mon * np.sqrt(f0)))-1 / np.sqrt(f0)
    f_D = optimize.newton(colebrook, 0.02)  #fator atrito de Darcy
    f_F = f_D / 4.                          #fator atrito de Fanning

    C1, C2, C3 = 0., -2 * Gt2 * spvolL * (f_F / D), -2 * Gt2 * spvolL * (f_F / D)
    
    
    matrizA = np.array([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])
    RHS_C = np.array([C1, C2, C3])
    dudz, dpdz, dhdz = np.linalg.solve(matrizA, RHS_C)
    return [dudz, dpdz, dhdz]



#[7] ==============FUNÇÃO A SER INTEGRADA =================
#source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html

#=================== CREATING VECTORS POINTS =============================
pointsNumber = 100    
Zduct = np.linspace(0, Ld, pointsNumber + 1)

#==================== BUILDING INITIAL VALUES ============
u_e = (mdotL_e * spvolL_e) / A                   # Para obter u_e (subcooled liquid)
F_V, h_e, s_e = hsFv_obj(p_e, T_e, xRe)       #Para obter h_e (subcooled liquid, so F_V = 0)
uph_0 = [u_e, p_e, h_e]



#===================== INTEGRATION ========================

uph_singlephase = integrate.odeint(systemEDOsinglePhase, uph_0, Zduct, args=(Gt, D, MMixture, viscL, ks, h_e, T_e, CpL))





# #[8] ================= TAKING THE RESULTS =================
u = uph_singlephase[:,0]
p = uph_singlephase[:,1]
h = uph_singlephase[:,2]
pB_v = pB * np.ones_like(Zduct)
alfa = twoPhaseFlowTools_obj.fracaoVazio(0.01, p,T_e,MMixture, spvolL_e)



# #[9]=========================== PLOT =====================

plt.figure(figsize=(7,5))
#plt.ylim(20,120)
plt.xlabel('Z [m]')
plt.ylabel('Void Fraction')
plt.plot(Zduct, alfa)
plt.legend(['Fração de Vazio'], loc=3)


plt.figure(figsize=(7,5))
#plt.ylim(20,120)
plt.xlabel('Z [m]')
plt.ylabel('u [m/s]')
plt.plot(Zduct, u)
plt.legend(['$u_{incompressivel}$ ao longo do duto'], loc=1) #loc=2 vai para canto sup esq


plt.figure(figsize=(7,5))
#plt.ylim(20,120)
plt.xlabel('Z [m]')
plt.ylabel('P [Pascal]')
plt.plot(Zduct, p)
plt.plot(Zduct, pB_v)
plt.legend(['Pressao Esc. Incompressivel', 'Pressão Saturação'], loc=3)

plt.figure(figsize=(7,5))
#plt.ylim(20,120)
plt.xlabel('Z [m]')
plt.ylabel('H [J/kg]')
plt.plot(Zduct, h)
plt.legend(['$h_{incompressivel}$ ao longo do duto'], loc=3)


plt.show()
plt.close('all')
