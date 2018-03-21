# -*- coding: utf-8 -*-
import numpy as np 
from scipy import integrate, optimize
import matplotlib.pyplot as plt
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
#P_sat: pressão de saturação [Pa] @ P_sat = P_sat(T_e)
#x_e: título de vapor entrada duto [-]
#Mt = Mg_e + Mf_e: vazão massica total (kg/s)
#D: diametro em inch transformado para metros [m]
#Ld: comprimento total do duto [m]
#deltaLd: quantidade divisoes do duto, ou seja, quantidade de Z steps [-]
#rks: ugosidade absoluta [m]
#teta: ângulo inclinação medido a partir da horizontal(+ para escoam. ascend.) [graus]
#rho_f: massa específica do líquido [kg/m3] @ [Tsat(P_amb)] - pg 532 - Ghiaasiaan
#vis_f: viscosidade do líquido [k/(m.s)] @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
#vis_g: viscosidade do gás [k/(m.s)] @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
#I: ascendente ou descendente de acordo com teta
#Gt: fluxo massico superficial total ((kg/s)/m2)
#rug = ks/D: rugosidade relativa [-]
#teta_rad: ângulo teta em radianos [rad]
#A: área transversal duto [m2]
#Sr = ug/ul: speed ratio
(P_e, P_sat, T_e, Mg_e, Mf_e, mw, vis_g, vis_f, rho_f) = input_flow_data_function()
(D, Ld, ks, teta) = input_pipe_data_function()
T = T_e                   
Sr = 1.


#[2]========== OUTRAS CONSTANTES + CÁLCULO SIMPLES ===========
A = np.pi * np.power(D, 2) / 4      
teta_rad = teta * np.pi / 180.      
rug = ks / D
Mt = Mg_e + Mf_e                           
Gt = Mt / A                                                  
deltaLd = 100                               
I = teta/np.absolute((teta + 1e-6)) 

#[3]================= CONDIÇÕES INICIAIS: j_init, x_init e P_init ===========
x_e = Mg_e / Mt                     #título de vapor na entrada [-]


def massaEspecificaGas(P, T, mw):
    return (P * mw / (R * T))

rho_g_e = massaEspecificaGas(P_e, T, mw)
j_e = (Mg_e / rho_g_e + Mf_e / rho_f) / A # m/s


def massaEspecificaBifasico(x, P, T, mw, rho_f):
    rho_g = massaEspecificaGas(P, T, mw)
    return np.power(((1.-x) / rho_f + x / rho_g),-1)

    
def fracaoVazio(x, P, T, mw, rho_f, Sr):  # Eq 3.39, pg 96 Ghiaasiaan
    rho_g = massaEspecificaGas(P, T, mw)
    return (rho_f * x / (rho_f * x + rho_g * Sr * (1.-x)))


#[4]==PROPRIEDADES DE TRANSPORTE 
def viscosidadeBifasica(x, P, T, mw, Sr, vis_g, vis_f, rho_f):
    alfa = fracaoVazio(x, P, T, mw, rho_f, Sr)
    return (alfa * vis_g + vis_f * (1. - alfa) * (1. + 2.5 * alfa))  #Eqc 8.33, pag 213 Ghiaasiaan


def reynoldsBifasico(Gt, D, x, P, T, mw, Sr, vis_g, vis_f, rho_f):
    vis_tp = viscosidadeBifasica(x, P, T, mw, Sr, vis_g, vis_f, rho_f)
    return (Gt * D / vis_tp)


#[5] == PROPRIEDADES CALOR & GÁS [água pura]
hg = 2675.7e3               # J/kg @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
hf = 419.06e3               # J/kg @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
dZdP = 0.                   #tx variacao fator compressibilidade do gás com a pressão
Cp_g = 2.029e3               # J/(kg K) @ [Tsat(P_amb)] - pg 532 - Ghiaasiaan
Cp_f = 4.217e3               # J/(kg K) @ [Tsat(P_amb)] - pg 532 - Ghiaasiaan
Cv_g = Cp_g - R / mw        #Cv=Cp-R; dividido por mw para passar R para base massica @ [Tsat(P_amb)]
Cv_f = Cp_f                 #Cv_f = Cp_f (aprox.)

def derivadadPdT (P, T, mw, rho_f): #Clapeyron - eqc (1.9) - pg 5 - Ghiaasiaan
    rho_g = massaEspecificaGas(P, T, mw)
    return (hg - hf) / (      T * (  np.power(rho_g,-1) - np.power(rho_f,-1) )        )


def derivadadrhodP (P, T, mw, rho_f): #ver exemplo (1.5) - pg 19 - Ghiaasiaan (adaptado subst. pura @ sat)
    dPdT = derivadadPdT(P, T, mw, rho_f)
    return ( mw / (R * T) - P * mw / (R * np.power(T,2)) * np.power(dPdT,-1) ) #preciso inverso Clapeyron


q_fluxo = 1e3            # J/(m2*s) fluxo de calor q"


#[6]============================ MAIN - ODE's system =========================================
#source: The EDO's system wrote here was based on page 167  Ghiaasiaan
# How to solve this system? See the page -->
# --> https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
def sistemaEDO(jxP, Z, Gt, D, T, mw, Sr, vis_g, vis_f, rho_f, ks):
    j, x, P = jxP
    rho_g = massaEspecificaGas(P, T, mw)
    rho_tp = massaEspecificaBifasico(x, P, T, mw, rho_f)
    Re_tp = reynoldsBifasico(Gt, D, x, P, T, mw, Sr, vis_g, vis_f, rho_f)
    print("Variável Sem Uso", Re_tp)
    Re_mon = Gt * D / vis_f
    dPdT = derivadadPdT(P, T, mw, rho_f)
    drho_gdP = derivadadrhodP(P, T, mw, rho_f)
    
    dhgdP = Cp_g * np.power(dPdT,-1)         #Regra da cadeia (dhgdP) = (dhg/dT)*(dT/dP) alimentada por Eq entalpia (pg 20)
                                                #  ...para obter (dhg/dT) e o inverso Clapeyron para obter (dT/dP)
    dhfdP = Cp_f * np.power(dPdT,-1)         #idem ao procedimento do vapor dhgdP
    
    A11 = rho_tp
    A12 = j * (1. / rho_f - 1. / rho_g) * np.power(rho_tp,2)
    A13 = j * drho_gdP * np.power(rho_tp,2) * x / np.power(rho_g,2)
    A21 = rho_tp * j
    A22 = 0.
    A23 = 1.
    A31 = 0.
    A32 = rho_tp * j * (hg - hf)
    A33 = rho_tp * j * ((1. - x) * dhfdP + x * dhgdP) - j
    
    
    colebrook = lambda f0 : 1.14 - 2. * np.log10(ks / D + 9.35 / (Re_mon * np.sqrt(f0)))-1 / np.sqrt(f0)
    fAtrito = optimize.newton(colebrook,0.02) #fator atrito de Darcy
    tau_w = (fAtrito / 4) * np.power(Gt,2) / ( 2 * rho_tp)
    print("Segunda Variável Sem Uso", tau_w)
    PHI_LO_2 = ( 1 + (vis_f - vis_g) * x / vis_g )**(-0.25) * ( 1 + (rho_f / rho_g - 1.) * x )


    C1 = 0.
    # C2 = (-I * rho_tp * g * np.sin(teta_rad) - 4 * tau_w / D)
    # C3 = 4 * q_fluxo / D + 4 * j * tau_w / D
    C2 = ( -I * rho_tp * g * np.sin(teta_rad) - PHI_LO_2 * (2 * fAtrito * np.power(Gt,2) / (D * rho_f) ))
    C3 = 4 * q_fluxo / D - np.power(rho_tp,-1) * PHI_LO_2 * (2 * fAtrito * np.power(Gt,2) / (D * rho_f) )
    
    #dPdZ = (-A11*C2/A21-A12*C3/A32+C1)/(A13-A11*A23/A21-A12*A33/A32)
    
    matrizA = np.array([[A11,A12,A13],[A21,A22,A23],[A31,A32,A33]])
    RHS_C = np.array([C1, C2, C3])
    djdz, dxdz, dPdz = np.linalg.solve(matrizA, RHS_C)
    return djdz, dxdz, dPdz 

def incompressivel(P, Z, Gt, D, vis_f, ks):
    Re_mon = Gt * D / vis_f
    colebrook = lambda f0 : 1.14 - 2. * np.log10( ks / D + 9.35 / (Re_mon * np.sqrt(f0)))-1 / np.sqrt(f0)
    fAtrito = optimize.newton(colebrook,0.02) #fator atrito de Darcy
    tau_w = (fAtrito / 4) * np.power(Gt,2) / (2 * rho_f)
    dPdZ = (- I * rho_f * g * np.sin(teta_rad) - 4 * tau_w / D)
    return dPdZ



#[7] ==============FUNÇÃO A SER INTEGRADA =================
#source: https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.odeint.html
Z = np.linspace(0, Ld, deltaLd + 1)
jxP_init = [j_e, x_e, P_e]
P_Incomp = integrate.odeint(incompressivel, P_e, Z, args=(Gt, D, vis_f, ks))
jxP_Comp = integrate.odeint(sistemaEDO, jxP_init, Z, args=(Gt, D, T, mw, Sr, vis_g, vis_f, rho_f, ks))



#[8] ============== EXTRAINDO RESULTADOS =================
j = jxP_Comp[:,0]
x = jxP_Comp[:,1]
P = jxP_Comp[:,2]
P_sat_v = P_sat * np.ones_like(Z)
alfa = fracaoVazio(x, P, T, mw, rho_f, Sr)
print("aqui o valor do título", x)


qtd_pontos = Z.shape[0]
for int in np.arange(0, qtd_pontos):
    dPdZ_incompressivel = incompressivel(P_Incomp[int], Z, Gt, D, vis_f, ks)
    var = j[int], x[int], P[int]
    resultado = sistemaEDO(var, Z, Gt, D, T, mw, Sr, vis_g, vis_f, rho_f, ks)
    print("dPdZ_comp = ", resultado[2])
    print("dPdZ_inc = ", dPdZ_incompressivel)
    print("interador", int)




# print("dPdZ_compressivel", resultado[2])
print("posicao Z", Z)
# print("pressao", P)
# print("titulo x", x)
# print("velocidade j", j)
#if (P_inc>Psat_v).all():
#    Pp=P_inc
#else:
#    Pp=jxP_integrado[:,2]
#
# if  (P > Psat_v).all():
#     Pp = P_inc
# else:
#     Pp = P #jxP_integrado[:,2]


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

#plt.figure(figsize=(7,5))
##plt.ylim(20,120)
#plt.xlabel('Comprimento z [m]')
#plt.ylabel('Velocidade Superficial j [m/s]')
#plt.plot(Z,j)
#plt.legend(['Velocidade Superficial (j) ao longo do duto'], loc=1) #loc=2 vai para canto sup esq

plt.figure(figsize=(7,5))
#plt.ylim(20,120)
plt.xlabel('Comprimento z [m]')
plt.ylabel('Pressao [Pascal]')
plt.plot(Z, P)
#plt.plot(Z,P)
plt.plot(Z, P_sat_v)
plt.plot(Z, P_Incomp)
plt.legend(['Pressao Esc. Compressivel', 'Pressão Saturação', 'Pressão Esc. Incompressível'], loc=3)

#rho_g=massaEspecificaGas(P,T)
#alfa=fracaoVazio(x,rho_g)
#
plt.figure(figsize=(7,5))
plt.ylim(0,1)
plt.xlabel('Comprimento z [m]')
plt.ylabel('Fração de Vazio e Título [-]')
plt.plot(Z,alfa)
plt.plot(Z,x)
plt.legend(['Fração de vazio', 'Título de vapor'], loc=1) #loc=2 vai para canto sup esq
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
