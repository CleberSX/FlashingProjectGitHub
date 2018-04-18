import numpy as np
from scipy.optimize import fsolve


from CoolProp.CoolProp import PropsSI
from InputData___ReadThisFile import props
import Tools_Convert
from FlowTools_file import FlowTools_class
from input_pipe_data import input_pipe_data_function as pipe
from input_pipe_data import areaVenturiPipe_function as Area
from input_flow_data import input_flow_data_function

''' 
ESTE ARQUIVO FOI CRIADO PARA SER APENAS UMA CALCULADORA, CUJO OBJETIVO É 
TESTAR O PELO MOLECULAR DO ÓLEO: POE ISO VG 10. ISSO FOI NECESSÁRIO PORQUE 
A VISCOSIDADE DA MISTURA NÃO "BATE" COM OS RESULTADOS PRESENTES NA TESE DE BERTOLDI.

O QUE O ARQUIVO FAZ? QUER ENCONTRAR O PESO MOLECULAR DO POE ISO VG 10 (isso porque na tese do Bertoldi 
não foi informado o valor do PMóleo)

OS TESTES FORAM COMPARADOS COM OS DA TABELA 4.1 (pg 80) UTILIZANDO A EQ. (4.1) E A EQ. (4.2). 
FOI REALIZADA UMA ENGENHARIA REVERSA, ONDE A VISCOSIDADE DA MISTURA, FRAÇÃO MÁSSICA, TEMPERATURA
E PRESSÃO FORAM OBTIDOS DA TABELA 4.1. EM SEGUIDA FOI UTILIZADO O fsolve PARA DETERMINAR A FRAÇÃO
MOLAR DE R134a QUE FAZ ZERAR A EQ. (4.1) \n


A VISCOSIDADE DO REFRIGERANTE R134a É OBTIDO VIA CoolProp

A VISCOSIDADE DO ÓLEO É CALCULADA VIA EQUAÇÃO DA TESE DO BERTOLDI  (page 82)
'''



##############################################################
#Testes...Discutir com Prof Jader

(pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props
LC, base = 0.01/100, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
xRe, xRe_mass = Tools_Convert.frac_input(MM, zin, base)



############################################################
#variables that are necessary to be imported 
(angleVenturi_in, angleVenturi_out, ks, Ld, D, Dvt, ziv, zig, zfg, zfv) = pipe() 
Ac_pipe = Area(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, 0.0)
(p_e, T_e, mdotL_e) = input_flow_data_function() 
Gt = mdotL_e / Ac_pipe

#creating the object
FlowTools_obj = FlowTools_class(D, Gt)


visc_mixture_in = FlowTools_obj.viscosidadeMonofasico(T_e, p_e, xRe)
print('minha viscosidade', visc_mixture_in)
print('percentual em massa refrigerante', LC)
print('vetor xRe massico', xRe_mass)
print('vetor xRe molar', xRe)


viscPOE_in = FlowTools_obj.viscO_function(T_e)
viscR_in = FlowTools_obj.viscR_function(T_e, p_e)
print('\n============================================================ \n')
print("\n Nas @ de entrada duto T_e [C] = ", (T_e - 273.15), " e p_e [1e5 Pa]", p_e / 1e5, "\n")
print("viscosidade R134a CoolProp @ entrada duto", viscR_in * 1e6 ,"[1e-6 Pa.s]")
print("viscosidade do óleo puro @ entrada duto", viscPOE_in * 1e6 ,"[1e-6 Pa.s]")
print('\n============================================================ \n')








#Análise Tabela 4.1, pg 80 Tese Dalton -- [Casos 1 ao 5]
#fração massica refrigerante 
fxR = np.array([16.1, 21.7, 30.1, 39.4, 46.7]) * 1e-2
MMR = 102.032 



#viscosidade da mistura resultante (6ª coluna)
mu_mix = np.array([6030, 4750, 3320, 2230, 1630], float) * 1e-6
ln_mu_mix = np.log(mu_mix)



#vetor pressão (5ª coluna)
p_vector = np.array([4.79, 5.8, 7.16, 7.96, 8.63]) * 1e5



#determinando viscosidade R134a via CoolProp
mu_r134a = np.zeros_like(mu_mix) 
for j, p in enumerate(p_vector):
    mu_r134a[j] = FlowTools_obj.viscosidadeMonofasico(T_e, p, [1.0, 0.0])
    #print('\n valor viscosidade R134a via CoolProp com p = ', p / 1e5, 
    #' [10^2 kPa]; T = ', (T_e-273.15), '[C]; \n visc [Pa.s] \n', mu_r134a, '\n')
ln_mu_r134a = np.log(mu_r134a)



#viscosidade POE ISO VG 10 (Eq. 4.2, pg 82)
mu_vg10 = np.ones_like(mu_mix) * viscPOE_in
ln_mu_vg10 = np.log(mu_vg10)



#escrevendo a função que representa a Eq. (4.1), pg 81
def getxRmolar(xR134a, ln_mu_mix, ln_mu_r134a, ln_mu_vg10, G12):
    '''
        Deseja-se determinar o valor de x1 que faz zerar a função, no 
         caso a Eq. (4.1), pg 81 \n
        xR134a: fração molar R134a \n
        xPOE: fração molar POE ISO VG 10 
      ''' 
    xPOE = 1. - xR134a
    return ln_mu_mix - (xR134a * ln_mu_r134a + xPOE * ln_mu_vg10 + xR134a * xPOE * G12 ) 



#resolvendo numéricamente para encontrar xR134a que zera a getxRmolar
G12 = 3.5
xR134a_initial_guess = np.ones_like(mu_mix) * 0.5
xR134a_molar_solution = fsolve(getxRmolar, xR134a_initial_guess, args=(ln_mu_mix, ln_mu_r134a, ln_mu_vg10, G12))
print('frações massicas R134a Tese Dalton (coluna 2, pg 80)', fxR)
print('respectivas frações MOLARES R134a [resolvendo Eq.4.1] ', np.around((xR134a_molar_solution), decimals=3))


#uma vez encontrada a concentração molar de R134a, determina-se o peso...
#...molecular do POE ISO VG 10
def calculaMo(xR_molar, xR_mass, PM_R134a):
    '''
    Tendo as frações mássicas, molares e peso molecular do R134a determina-se
        ...o peso molecular do POE ISO VG 10 \n
    xR_molar: fração molar de R134a \n
    xR_mass: fração mássica de R134a \n
    PM_R134a: peso molecular do R134a 
      '''
    return PM_R134a * (1.-xR_mass) * xR_molar / ((1.-xR_molar) * xR_mass)



PesoMolecularPOE_ISO_VG10 = calculaMo(xR134a_molar_solution, fxR, MMR)
print('peso molecular POE ISO VG 10 - engenharia reversa', np.around((PesoMolecularPOE_ISO_VG10), decimals=2))
