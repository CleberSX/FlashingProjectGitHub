# -*- coding: utf-8 -*-
import numpy as np 
from scipy import integrate, optimize
import matplotlib.pyplot as plt
import Tools_Convert
from InputData___ReadThisFile import props
from BubbleP import bubble_obj
from input_flow_data import input_flow_data_function
from input_pipe_data import input_pipe_data_function

#Constantes
g = 9.8                             # m/s^2
R = 8314.34                         # J / (kmol K)


"""
#======================== PARAMETROS DE ENTRADA (O PROGRAMA TODO no SI) ======================
"""
#[1]================= DADOS DE ENTRADA ================= 
#Mg_e: vazao massica de vapor entrada duto [kg/s]
#Mf_e: vazao massica de líquido entrada duto [kg/s]
#mw: molecular weight - peso molecular do vapor água [kg/kmol]
#T_e: temperatura de entrada [K]
#P_e: pressao de entrada [Pa]
#pB: pressão de saturação [Pa] @ pB = pB(T_e)  
#x_e: título de vapor entrada duto [-]
#Mt = Mg_e + Mf_e: vazão massica total (kg/s)
#D: diametro em inch transformado para metros [m]
#Ld: comprimento total do duto [m]
#deltaLd: quantidade divisoes do duto, ou seja, quantidade de Z steps [-]
#rks: ugosidade absoluta [m]
#teta: ângulo inclinação medido a partir da horizontal(+ para escoam. ascend.) [graus]
#rho_f: massa específica do líquido [kg/m3] @ [Tsat(P_amb)] - pg 532 - Ghiaasiaan
#vef_e = 1/rho_f: volume específico do líquido na entrada [m3/kg]
#veg_e: volume específico do gás na entrada [m3/kg]
#vis_f: viscosidade do líquido [k/(m.s)] @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
#vis_g: viscosidade do gás [k/(m.s)] @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
#I: ascendente ou descendente de acordo com teta
#Gt: fluxo massico superficial total ((kg/s)/m2)
#rug = ks/D: rugosidade relativa [-]
#teta_rad: ângulo teta em radianos [rad]
#A: área transversal duto [m2]
#Sr = ug/ul: speed ratio

'''
=================================================================================
TAKING SPECIFIC HEAT FROM props (LOOK AT THE HEAD)
=================================================================================
'''
(pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props

'''
=================================================================================
TAKING SATURATION PRESSURE FROM BubbleP.py
FOR MORE INFORMATION ABOUT BubbleP consult this BubbleP.py
=================================================================================
'''
(P_e, T_e, Mg_e, Mf_e, mw, vis_g, vis_f, rho_f) = input_flow_data_function()
(D, Ld, ks, teta) = input_pipe_data_function()

T = T_e
LC, base = 99./100, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
z, z_mass = Tools_Convert.frac_input(MM, zin, base)
pG = 1.2 * bubble_obj.pressure_guess(T, z)
pB, y, Sy, counter = bubble_obj(T, z)
y_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, y)




T = T_e                   
Sr = 1.
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
Mt = Mg_e + Mf_e                           
Gt = Mt / A  
vef_e = np.power(rho_f,-1)                                                
deltaLd = 100                               
I = teta/np.absolute((teta + 1e-6)) 

#[3]================= CONDIÇÕES INICIAIS: j_init, x_init e P_init ===========
x_e = Mg_e / Mt                     #título de vapor na entrada [-]


def volumeEspecificoGas(P, T, mw):
    return ( R * T / (P * mw))

veg_e = volumeEspecificoGas(P_e, T, mw)
j_e = (Mg_e * veg_e + Mf_e * vef_e) / A # m/s


def volumeEspecificaBifasico(x, P, T, mw, vef_e):
    veg = volumeEspecificoGas(P, T, mw)
    return ((1.-x) * vef_e + x * veg)

    
def fracaoVazio(x, P, T, mw, vef_e):  # Eq 3.19, pg 37 Tese
    veg = volumeEspecificoGas(P, T, mw)
    veb = volumeEspecificaBifasico(x, P, T, mw, vef_e)
    return (veg * x / veb)


#[4]==PROPRIEDADES DE TRANSPORTE 
def viscosidadeBifasica(x, P, T, mw, vis_g, vis_f, vef_e):
    alfa = fracaoVazio(x, P, T, mw, vef_e)
    return (alfa * vis_g + vis_f * (1. - alfa) * (1. + 2.5 * alfa))  #Eqc 8.33, pag 213 Ghiaasiaan


def reynoldsBifasico(Gt, D, x, P, T, mw, vis_g, vis_f, vef_e):
    vis_tp = viscosidadeBifasica(x, P, T, mw, vis_g, vis_f, vef_e)
    return (Gt * D / vis_tp)


#[5] == PROPRIEDADES CALOR & GÁS [água pura]
hg = 2675.7e3                       # J/kg @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
hf = 419.06e3                       # J/kg @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
dZdP = 0.                           # tx variacao fator compressibilidade do gás com a pressão
Cp_g = 2.029e3                      # J/(kg K) @ [Tsat(P_amb)] - pg 532 - Ghiaasiaan
Cp_f = np.einsum('i,i', z, Cp)      # J/(kg K) @ solução ideal (dados de Cp vindos de props)
T_ref = 273.15                      # T_ref temperatura de referência [K]
h_ref = 0.0                         # entalpia de referência  [J/kg] @ T_ref
h_e = Cp_f * (T_e  - T_ref) + h_ref #Tenho que analisar melhor (considerei T* = 273.15K e h* = 0)
Cv_g = Cp_g - R / mw                #Cv=Cp-R; dividido por mw para passar R para base massica @ [Tsat(P_amb)]
Cv_f = Cp_f                         #Cv_f = Cp_f (aprox.)



#[6]============================ MAIN - ODE's system =========================================
#source: The EDO's system wrote here was based on page 167  Ghiaasiaan
# How to solve this system? See the page -->
# --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
def sistemaEDO(jPh, Z, Gt, D, T, mw, vis_f, vef_e, ks):
    j, P, h = jPh
    Re_mon = Gt * D / vis_f
    
    A11 = np.power(j,-1)
    A12 = 0.
    A13 = 1.e-5
    A21 = j
    A22 = vef_e
    A23 = 0.
    A31 = j
    A32 = 0.
    A33 = 1.
    
    
    colebrook = lambda f0 : 1.14 - 2. * np.log10(ks / D + 9.35 / (Re_mon * np.sqrt(f0)))-1 / np.sqrt(f0)
    fAtrito = optimize.newton(colebrook, 0.02) #fator atrito de Darcy
    f_F = fAtrito / 4.

    C1 = 0.
    C2 = -2 * np.power(Gt,2) * vef_e * (f_F / D)
    C3 = C2
    
    
    matrizA = np.array([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])
    RHS_C = np.array([C1, C2, C3])
    djdz, dPdz, dhdz = np.linalg.solve(matrizA, RHS_C)
    return djdz, dPdz, dhdz 



#[7] ==============FUNÇÃO A SER INTEGRADA =================
#source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
Z = np.linspace(0, Ld, deltaLd + 1)
jPh_init = j_e, P_e, h_e
jPh_singlephase = integrate.odeint(sistemaEDO, jPh_init, Z, args=(Gt, D, T, mw, vis_f, vef_e, ks))



#[8] ============== EXTRAINDO RESULTADOS =================
j = jPh_singlephase[:,0]
P = jPh_singlephase[:,1]
h = jPh_singlephase[:,2]
pB_v = pB * np.ones_like(Z)


# qtd_pontos = Z.shape[0]
# for int in np.arange(0, qtd_pontos):
#     var = j[int], P[int], h[int] 
#     resultado = sistemaEDO(var, Z, Gt, D, T, mw, vis_f, vef_e, ks)
#     print("dPdZ_singlephase = ", resultado[2])
#     print("interador", int)





#[8]=========================== GRÁFICOS =====================
#plt.figure(figsize=(7,5))
##plt.ylim(20,120)
#plt.xlabel('Comprimento z [m]')
#plt.ylabel('Titulo [-]')
#plt.plot(Z,x)
#plt.legend(['Titulo ao longo do duto'], loc=1) #loc=2 vai para canto sup esq


#plt.figure(figsize=(7,5))
##plt.ylim(20,120)
#plt.xlabel('Comprimento z [m]')
#plt.ylabel('Pressao [Pascal]')
#plt.plot(Z,P)
#plt.legend(['Pressao ao longo do duto'], loc=1) #loc=2 vai para canto sup esq
msg = "Foi feito o cálculo do Cp_f do single-phase liquid"
print(msg)
print(Cp_f)

plt.figure(figsize=(7,5))
#plt.ylim(20,120)
plt.xlabel('Z [m]')
plt.ylabel('u [m/s]')
plt.plot(Z,j)
plt.legend(['$u_{incompressivel}$ ao longo do duto'], loc=1) #loc=2 vai para canto sup esq


plt.figure(figsize=(7,5))
#plt.ylim(20,120)
plt.xlabel('Z [m]')
plt.ylabel('P [Pascal]')
plt.plot(Z, P)
plt.plot(Z, pB_v)
plt.legend(['Pressao Esc. Incompressivel', 'Pressão Saturação'], loc=3)

plt.figure(figsize=(7,5))
#plt.ylim(20,120)
plt.xlabel('Z [m]')
plt.ylabel('H [J/kg]')
plt.plot(Z, h)
plt.legend(['$h_{incompressivel}$ ao longo do duto'], loc=3)

#rho_g=massaEspecificaGas(P,T)
#alfa=fracaoVazio(x,rho_g)
#
# plt.figure(figsize=(7,5))
# plt.ylim(0,1)
# plt.xlabel('Comprimento z [m]')
# plt.ylabel('Fração de Vazio e Título [-]')
# plt.plot(Z,alfa)
# plt.plot(Z,x)
# plt.legend(['Fração de vazio', 'Título de vapor'], loc=1) #loc=2 vai para canto sup esq
#
# rho_g = massaEspecificaGas(P,T_e, mw)
# plt.figure(figsize=(7,5))
# plt.xlabel('Comprimento z [m]')
# plt.ylabel('Massa Específica do Gás [kg/m3]')
# plt.plot(Z,rho_g)
# plt.legend(['Massa Específica do Gás ao longo do duto'], loc=1) #loc=2 vai para canto sup esq
#
#
#
#Mg = vazaoMassicaGas(x)
#plt.figure(figsize=(7,5))
##plt.ylim(20,120)
#plt.xlabel('Comprimento z [m]')
#plt.ylabel('Vazão mássica de gás [kg/s]')
#plt.plot(Z,Mg)
#plt.legend(['Vazão mássica de gás ao longo do duto'], loc=1) #loc=2 vai para canto sup esq
#
plt.show()
plt.close('all')
