# -*- coding: utf-8 -*-
import numpy as np 
from scipy import integrate, optimize
import matplotlib.pyplot as plt
import Tools_Convert
from InputData___ReadThisFile import props
from BubbleP import bubble_obj
from Properties import Properties
from EOS_PengRobinson import PengRobinsonEos
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
#mdotg_e: taxa (tx) massica de vapor entrada duto [kg/s]
#mdotf_e: tx massica de líquido entrada duto [kg/s]
#MMixture: molecular weight - peso molecular do vapor água [kg/kmol]
#T_e: temperatura de entrada [K]
#p_e: pressao de entrada [Pa]
#pB: pressão de saturação [Pa] @ pB = pB(T_e)
#pB_v: é o próprio pB, porém na forma vetorial 
#x_e: título de vapor entrada duto [-]
#mdott = mdotg_e + mdotf_e: tx massica total (kg/s)
#D: diametro em inch transformado para metros [m]
#Ld: comprimento total do duto [m]
#deltaLd: quantidade divisoes do duto, ou seja, quantidade de Z steps [-]
#rks: ugosidade absoluta [m]
#teta: ângulo inclinação medido a partir da horizontal(+ para escoam. ascend.) [graus]
#rhog: massa específica do gás [kg/m3] 
#rhof: massa específica do líquido [kg/m3] 
#rhof_e: massa específica do líquido na entrada [kg/m3]
#rhog_e: massa específica do gás na entrada [kg/m3] 
#vef_e = 1/rhof_e: volume específico do líquido na entrada [m3/kg] 
#veg_e: volume específico do gás na entrada [m3/kg]
#vis_f: viscosidade do líquido [k/(m.s)] 
#vis_g: viscosidade do gás [k/(m.s)] 
#I: ascendente ou descendente de acordo com teta
#Gt: fluxo massico superficial total ((kg/s)/m2)
#rug = ks/D: rugosidade relativa [-]
#teta_rad: ângulo teta em radianos [rad]
#A: área transversal duto [m2]
#u: flow speed [m/s]
#ug: gas speed [m/s]
#ul: liquid speed [m/s]
#Sr = ug/ul: speed ratio
#LC: light component (our refrigerant)
#Z: compressibility factor --> [p*veg = (Z*R*T) / MM]
#z: binary mixture composition at entrance: z = ([refrigerant],[oil]) where [brackets] means concentration - [kg Refrig / kg mixture]
#Zduct: duct length - [m]
#f_D: Darcy friction factor
#f_F: Fanning friction factor


'''
=================================================================================
TAKING SPECIFIC HEAT FROM props (LOOK AT THE HEAD)
=================================================================================
'''
(pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props


'''
=================================================================================================================
NECESSARY OBJECTS
=================================================================================================================
'''
prop_obj = Properties(pC, Tc, AcF, omega_a, omega_b, kij)
eos_obj = PengRobinsonEos(pC, Tc, AcF, omega_a, omega_b, kij)

'''
=================================================================================
OTHERS INPUT DATA: FLOW & DUCT
=================================================================================
'''
(p_e, T_e, mdotg_e, mdotf_e, vis_g, vis_f) = input_flow_data_function()
(D, Ld, ks, teta) = input_pipe_data_function()


'''
=================================================================================
TAKING SATURATION PRESSURE FROM BubbleP.py
FOR MORE INFORMATION ABOUT BubbleP consult this BubbleP.py
=================================================================================
'''
T = T_e
LC, base = 99./100, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
z, z_mass = Tools_Convert.frac_input(MM, zin, base)
pG = 1.2 * bubble_obj.pressure_guess(T_e, z)
pB, y, Sy, counter = bubble_obj(T_e, z)
y_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, y)


MMixture = prop_obj.calculate_weight_molar_mixture(MM, z,"liquid")



T = T_e                   
if __name__== '__main__':
    print('\n---------------------------------------------------')
    print('[1] - Guess pB [Pa]= %.8e' % pG)
    print('[2] - ======> at T = %.2f [C], pB = %.8e [Pa] ' % ((T - 273.15), pB) + '\n')
    print('[3] - Concentration vapor phase [molar] = ', y.round(3))
    print('[4] - Concentration vapor phase [mass] = ', y_mass.round(3))
    print('[5] - Pay attention if Sy is close to unity (Sy = %.10f) [molar]' % Sy)
    print('[6] - Global {mass} fraction = ', z_mass.round(3))
    print('[7] - Global {molar} fraction = ', z.round(3))



#[2]========== OUTRAS CONSTANTES + CÁLCULO SIMPLES ===========
A = np.pi * np.power(D, 2) / 4      
teta_rad = teta * np.pi / 180.      
rug = ks / D
mdott = mdotg_e + mdotf_e                           
Gt = mdott / A  
rhof_e = prop_obj.calculate_density_phase(p_e, T_e, MM, z, "liquid") 
vef_e = np.power(rhof_e,-1)                                                
deltaLd = 100                               
I = teta/np.absolute((teta + 1e-6)) 

#[3]================= CONDIÇÕES INICIAIS: j_init, x_init e P_init ===========
x_e = mdotg_e / mdott                     #título de vapor na entrada [-]


def volumeEspecificoGas(p, T, MMixture):
    return ( R * T / (p * MMixture))

veg_e = volumeEspecificoGas(p_e, T_e, MMixture)
u_e = (mdotg_e * veg_e + mdotf_e * vef_e) / A # m/s


def volumeEspecificaBifasico(x, p, T, MMixture, vef_e):
    veg = volumeEspecificoGas(p, T, MMixture)
    return ((1.-x) * vef_e + x * veg)

    
def fracaoVazio(x, p, T, MMixture, vef_e):  # Eq 3.19, pg 37 Tese
    veg = volumeEspecificoGas(p, T, MMixture)
    veb = volumeEspecificaBifasico(x, p, T, MMixture, vef_e)
    return (veg * x / veb)


#[4]==PROPRIEDADES DE TRANSPORTE 
def viscosidadeBifasica(x, p, T, MMixture, vis_g, vis_f, vef_e):
    alfa = fracaoVazio(x, p, T, MMixture, vef_e)
    return (alfa * vis_g + vis_f * (1. - alfa) * (1. + 2.5 * alfa))  #Eqc 8.33, pag 213 Ghiaasiaan


def reynoldsBifasico(Gt, D, x, p, T, MMixture, vis_g, vis_f, vef_e):
    vis_tp = viscosidadeBifasica(x, p, T, MMixture, vis_g, vis_f, vef_e)
    return (Gt * D / vis_tp)


#[5] == PROPRIEDADES CALOR & GÁS [água pura]
hg = 2675.7e3                       # J/kg @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
hf = 419.06e3                       # J/kg @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
dZdp = 0.                           # tx variacao fator compressibilidade do gás com a pressão
Cp_g = 2.029e3                      # J/(kg K) @ [Tsat(P_amb)] - pg 532 - Ghiaasiaan
Cp_f = np.einsum('i,i', z, Cp)      # J/(kg K) @ solução ideal (dados de Cp vindos de props)
T_ref = 273.15                      # T_ref temperatura de referência [K]
h_ref = 0.0                         # entalpia de referência  [J/kg] @ T_ref
h_e = Cp_f * (T_e  - T_ref) + h_ref # Tenho que analisar melhor (considerei T* = 273.15K e h* = 0)
Cv_g = Cp_g - R / MMixture          # Cv=Cp-R; dividido por MMixture para passar R para base massica @ [Tsat(P_amb)]
Cv_f = Cp_f                         # Cv_f = Cp_f (aprox.)



#[6]============================ MAIN - ODE's system =========================================
#source: The EDO's system wrote here was based on page 167  Ghiaasiaan
# How to solve this system? See the page -->
# --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
def systemEDOsinglePhase(uph, Zduct, Gt, D, T, MMixture, vis_f, ks):
    u, p, h = uph
    Gt2 = np.power(Gt, 2)
    Re_mon = Gt * D / vis_f
    rhof = prop_obj.calculate_density_phase(p, T, MM, z, "liquid")
    vef = np.power(rhof, -1)

    A11, A12, A13 = np.power(u,-1), 0., 1.e-5    
    A21, A22, A23 = u, vef, 0.
    A31, A32, A33 = u, 0., 1.
        
    colebrook = lambda f0 : 1.14 - 2. * np.log10(ks / D + 9.35 / (Re_mon * np.sqrt(f0)))-1 / np.sqrt(f0)
    f_D = optimize.newton(colebrook, 0.02)  #fator atrito de Darcy
    f_F = f_D / 4.                          #fator atrito de Fanning

    C1, C2, C3 = 0., -2 * Gt2 * vef * (f_F / D), -2 * Gt2 * vef * (f_F / D)
    
    
    matrizA = np.array([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])
    RHS_C = np.array([C1, C2, C3])
    dudz, dpdz, dhdz = np.linalg.solve(matrizA, RHS_C)
    return dudz, dpdz, dhdz 



#[7] ==============FUNÇÃO A SER INTEGRADA =================
#source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
Zduct = np.linspace(0, Ld, deltaLd + 1)
uph_init = u_e, p_e, h_e
uph_singlephase = integrate.odeint(systemEDOsinglePhase, uph_init, Zduct, args=(Gt, D, T, MMixture, vis_f, ks))



#[8] ============== EXTRAINDO RESULTADOS =================
u = uph_singlephase[:,0]
p = uph_singlephase[:,1]
h = uph_singlephase[:,2]
pB_v = pB * np.ones_like(Zduct)


qtd_pontos = Zduct.shape[0]
for int in np.arange(0, qtd_pontos):
    var = u[int], p[int], h[int] 
    resultado = systemEDOsinglePhase(var, Zduct, Gt, D, T, MMixture, vis_f, ks)
    rhof = prop_obj.calculate_density_phase(p[int], T, MM, z, "liquid")
    Tresult = T_ref + (h[int] - h_ref) / Cp_f
    print("Interactor = %i, dPdZ_singlephase = %.2f, Liquid Density = %.2f" % (int, resultado[2], rhof))
    print("Temperatura do escoamento", Tresult)




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

#rho_g=massaEspecificaGas(P,T)
#alfa=fracaoVazio(x,rho_g)
#
# plt.figure(figsize=(7,5))
# plt.ylim(0,1)
# plt.xlabel('Comprimento z [m]')
# plt.ylabel('Fração de Vazio e Título [-]')
# plt.plot(Zduct,alfa)
# plt.plot(Zduct,x)
# plt.legend(['Fração de vazio', 'Título de vapor'], loc=1) #loc=2 vai para canto sup esq
plt.show()
plt.close('all')
