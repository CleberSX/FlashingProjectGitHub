import numpy as np
import sys
from scipy.optimize import fsolve

from CoolProp.CoolProp import PropsSI
from InputData___ReadThisFile import props
import Tools_Convert

def input_flow_data_function():
    '''
    p_e: pipe's entrance pressure [Pa]
    T_e: pipe's entrance temperature  [K]  
    mdotL_e: pipe's entrance liquid mass mass ratio [kg/s]
    viscG: saturated gas's dynamic viscosity [kg/(m.s)]
    viscO: lubricant oil's dynamic viscosity [kg/(m.s)]
    viscR: refrigerant's dynamic viscosity [kg/(m.s)]   
    '''
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name)
    print('\n---\n')

    room_pressure = 1.e5                        # -- pressao ambiente [Pa]
    temperature = 309.8                         # -- temperatura de entrada [K]
    pressure = 8.81 * room_pressure              # -- pressão entrada duto [Pa]
    viscG = 12.02e-6                            # -- (@ saturated gas) [Pa.s]
    viscR = PropsSI("V", "T", temperature, "P", room_pressure,"R134a")    # -- (liquid refrigerant's dynamic viscosity ) [Pa.s]
    viscO =  viscO_function(temperature, 'K')   # -- (@ oil's dynamic viscosity ) [Pa.s]
    subcooled_liquid_mass_flow = 0.10           # -- (mass ratio subcooled liquid) [kg/s]

    return (pressure, temperature, subcooled_liquid_mass_flow, viscG, viscR, viscO)



def viscO_function(T, scale):
    '''
    This function calculate the POE ISO VG 10 viscosity \n
    
    You must give 2 arguments when call the function: (T,'C') or (T,'K') \n
    Oil Viscosity: in [Pa.s] \n


    This correlation has been gotten from: \n 
    Tese de doutorado do Dalton Bertoldi (2014), page 82 \n

    "Investigação Experimental de Escoamentos Bifásicos com mudança \n
    de fase de uma mistura binária em tubo de Venturi" \n
     '''
    if scale == 'K':
        Tcelsius = T - 273.15
    elif scale == 'C':
        Tcelsius = T 
    else:
        msg = 'You must specify what it\'s the temperature scale:'
        msg += ' "C" for Celsius or "K" for Kelvin'
        raise Exception(msg)
        

    return 0.04342 * np.exp(- 0.03529 * Tcelsius)




##############################################################
#Testes...Discutir com Prof Jader

(pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props
LC, base = 46.7/100, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
xRe, xRe_mass = Tools_Convert.frac_input(MM, zin, base)


T = (36.8 + 273.15)
p = 4.79e5
G12 = 3.5
viscO = viscO_function(T, 'K') 
viscR = PropsSI("V", "T", T, "P", p, "R134a") 
logvisc = np.array([np.log(viscR),np.log(viscO)])
sum_xlogvisc = np.einsum('i,i', xRe, logvisc)
xRxO_G12 = np.prod(xRe) * G12
resultado = np.exp(sum_xlogvisc + xRxO_G12) * 1e6
print('minha viscosidade', resultado)
print('percentual em massa refrigerante', LC)
print('vetor xRe massico', xRe_mass)
print('vetor xRe molar', xRe)


viscR_CoolProp = PropsSI("V", "T", T, "P", p,"R134a")
print("viscosidade R134a CoolProp", viscR_CoolProp ,"Pa.s", "T [K] = ", T, "Pressão [Pa]", p)


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
    mu_r134a[j] = PropsSI("V", "T", T, "P", p,"R134a")
print('\n valor viscosidade R134a via CoolProp com p = ', p / 1e5, 
    ' [10^2 kPa]; T = ', (T-273.15), '[C]; \n visc [Pa.s]', mu_r134a, '\n')
ln_mu_r134a = np.log(mu_r134a)

#viscosidade POE ISO VG 10 (Eq. 4.2, pg 82)
mu_vg10 = np.ones_like(mu_mix) * viscO_function(T, 'K')
ln_mu_vg10 = np.log(mu_vg10)

#escrevendo a função que representa a Eq. (4.1), pg 81
def getxRmolar(xR134a, ln_mu_mix, ln_mu_r134a, ln_mu_vg10, G12):
    '''
        Deseja-se determinar o valor de x1 que faz zerar a função, no 
         caso a Eq. (4.1), pg 81 \n
        x1: fração molar R134a \n
        x2: fração molar POE ISO VG 10 
      ''' 
    xPOE = 1. - xR134a
    return ln_mu_mix - (xR134a * ln_mu_r134a + xPOE * ln_mu_vg10 + xR134a * xPOE * G12 ) 


#resolvendo numéricamente para encontrar x1 que zera a getxRmolar
xR134a_initial_guess = np.ones_like(mu_mix) * 0.5
xR134a_molar_solution = fsolve(getxRmolar, xR134a_initial_guess, args=(ln_mu_mix, ln_mu_r134a, ln_mu_vg10, G12))
print('frações molares para CASOS 1 ao 5 Dalton (pg 80) \n', xR134a_molar_solution)

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
print(PesoMolecularPOE_ISO_VG10)
