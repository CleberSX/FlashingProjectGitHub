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
#deltaLd: quantidade divisoes do duto, ou seja, quantidade de Z steps [-]
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
#z_e: binary mixture composition at entrance: z_e = ([refrigerant]_e,[oil]_e) where [brackets]_e means concentration at entrance - [kg Refrig / kg mixture]_e
#z: binary mixture composition at some positon: z = ([refrigerant],[oil]) 
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


'''
=================================================================================
TAKING SATURATION PRESSURE FROM BubbleP.py
FOR MORE INFORMATION ABOUT Bubble Pressure consult BubbleP.py
=================================================================================
'''

LC, base = 99./100, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
z_e, z_mass_e = Tools_Convert.frac_input(MM, zin, base)
hR = hR_mass * prop_obj.calculate_weight_molar_mixture(MM, z_e, 'saturated_liquid')
sR = sR_mass * prop_obj.calculate_weight_molar_mixture(MM, z_e, 'saturated_liquid')
pG = 1.2 * bubble_obj.pressure_guess(T_e, z_e)
pB, y, Sy, counter = bubble_obj(T_e, z_e)
y_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, y)
MMixture = prop_obj.calculate_weight_molar_mixture(MM, z_e,"liquid")

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
    print('[6] - Feed global {mass} fraction = ', z_mass_e.round(3))
    print('[7] - Feed global {molar} fraction = ', z_e.round(3))



#[2]========== OUTRAS CONSTANTES + CÁLCULO SIMPLES ===========
A = np.pi * np.power(D, 2) / 4      
rug = ks / D
Gt = mdotL_e / A  
densL_e = prop_obj.calculate_density_phase(p_e, T_e, MM, z_e, "liquid") 
spvolL_e = np.power(densL_e,-1) 
viscL = viscF        #<--- considerando que a visc do líquido subresfriado é igual ao do líquido saturado                                   
deltaLd = 100                               


#[3]================= CONDIÇÕES INICIAIS: u_init, x_init e p_init ===========



def volumeEspecificoGas(p, T, MMixture):
    return ( R * T / (p * MMixture))


def volumeEspecificaBifasico(x, p, T, MMixture, spvolF):
    spvolG = volumeEspecificoGas(p, T, MMixture)
    return ((1.-x) * spvolF + x * spvolG)

    
def fracaoVazio(x, p, T, MMixture, spvolF):  # Eq 3.19, pg 37 Tese
    spvolG = volumeEspecificoGas(p, T, MMixture)
    spvolTP = volumeEspecificaBifasico(x, p, T, MMixture, spvolF)
    return (spvolG * x / spvolTP)


#[4]==PROPRIEDADES DE TRANSPORTE 
def viscosidadeBifasica(x, p, T, MMixture, viscG, viscF, spvolF):
    alfa = fracaoVazio(x, p, T, MMixture, spvolF)
    return (alfa * viscG + viscF * (1. - alfa) * (1. + 2.5 * alfa))  #Eqc 8.33, pag 213 Ghiaasiaan


def reynoldsBifasico(Gt, D, x, p, T, MMixture, viscG, viscF, spvolF):
    viscTP = viscosidadeBifasica(x, p, T, MMixture, viscG, viscF, spvolF)
    return (Gt * D / viscTP)


#[5] == PROPRIEDADES CALOR & GÁS 
CpL = np.einsum('i,i', z_e, Cp)     # -- capacidade térmica líquido subresfriado [J/(kg K)] (@ solução ideal)





#[6]============================ MAIN - ODE's system =========================================
#source: The EDO's system wrote here was based on page 167  Ghiaasiaan
# How to solve this system? See the page -->
# --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
def systemEDOsinglePhase(uph, Zduct, Gt, D, T, MMixture, viscL, ks):
    u, p, h = uph
    Gt2 = np.power(Gt, 2)
    Re_mon = Gt * D / viscL   
    densL = prop_obj.calculate_density_phase(p, T, MM, z_e, "liquid")
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
    return dudz, dpdz, dhdz 



#[7] ==============FUNÇÃO A SER INTEGRADA =================
#source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
Zduct = np.linspace(0, Ld, deltaLd + 1)
u_e = (mdotL_e * spvolL_e) / A                   # Para obter u_e (subcooled liquid)
F_V, h_e, s_e = hsFv_obj(p_e, T_e, z_e)       #Para obter h_e (subcooled liquid, so F_V = 0)



uph_init = u_e, p_e, h_e
uph_singlephase = integrate.odeint(systemEDOsinglePhase, uph_init, Zduct, args=(Gt, D, T_e, MMixture, viscL, ks))



#[8] ============== EXTRAINDO RESULTADOS =================
u = uph_singlephase[:,0]
p = uph_singlephase[:,1]
h = uph_singlephase[:,2]
pB_v = pB * np.ones_like(Zduct)


qtd_pontos = Zduct.shape[0]
for int in np.arange(0, qtd_pontos):
    var = u[int], p[int], h[int] 
    resultado = systemEDOsinglePhase(var, Zduct, Gt, D, T_e, MMixture, viscL, ks)
    densL = prop_obj.calculate_density_phase(p[int], T_e, MM, z_e, "liquid")
    T = TR + (h[int] - hR_mass) / CpL
    #print("Interactor = %i, dPdZ_singlephase = %.2f, Liquid Density = %.2f" % (int, resultado[2], densL))
    print("Temperatura do escoamento", T)




#[8]=========================== GRÁFICOS =====================

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
