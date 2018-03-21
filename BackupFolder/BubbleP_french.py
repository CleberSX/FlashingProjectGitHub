
import logging
import numpy as np
from scipy.optimize import newton, fsolve, root, brentq, brenth, brent, brute
from Wilson import calculate_K_values_wilson
from EOS_PengRobinson import PengRobinsonEos
from Michelsen import Michelsen
import Tools_Convert
from InputData___ReadThisFile import props
R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)


'''
=================================================================================================================
ESTA FUNCAO CALCULA A PRESSAO DE BOLHA.
PONTOS QUE DEVEMOS SE ATENTAR:
(1) - verificar qual arquivo (dados do problema) está sendo importado no arquivo ==> InputData__ReadThisFile.py
(2) - para calcular um ponto de bolha para um único ponto de temperatura e concetracao ==> vá para...
      ... BubbleP__singlePoint.py 
(3) - para levantar uma curva inteira de pontos bolha para varias temperatura e varias concetracoes ==> vá para...
      ... BubbleP__curvePoints.py


LEGEND:
AcF: Acentric factor [-]
omega_a, omega_b: Parameters from Peng Robinson (EdE)
kij: Binary interaction factor
Cp: Specific heat in molar base [J/kmol K]
T: Temperature [K]
Tc: Critical temperature [K]
TR: Reference temperature [K]
p: Pressure [Pa]
pC: Critical pressure [Pa]
pR: Reference pressure [Pa]
props: Just a name has been given to the properties list
z: global molar fraction
pG: Guess pressure [Pa] - It is necessary to initialize the search for correct bubble pressure  
pL and pH: Low pressure and High pressure [Pa] - They are necessary to build a window/range of pressure for the...
           ...numerical method begin the search the correct bubble pressure in Bubble.py
step: This is the step used by the numerical method in Bubble.py (very important)
pB: Bubble pressure [Pa]   
Sy: Sum of molar fraction vapor phase [-]
yi: Molar fraction of vapor phase [-]
================================================================================================================= 
'''


'''
=================================================================================================================
TO LOGGING MSG(import logging and logging.basicConfig and logging.disable). 
1st option: write on the screen
2nd option: write on the file

To use this tool you must choose uncomment one of the following options: the 1st OR the 2nd 
=================================================================================================================
'''

#1st OPTION:
#logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
#2nd OPTION
#logging.basicConfig(filename='Cleber_File.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.disable(logging.CRITICAL)


'''
=================================================================================================================
PROPS IS A LIST OF PROPERTIES; BLANK SPACE CORRESPOND TO THE PROPERTY WE ARE INTERESTED. 

CHECK BubbleP__singlePoint.py OR BubbleP__curvePoints.py to see how the empty spaces are filled
=================================================================================================================
'''
(_, _, _, pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props



'''
=================================================================================================================
NECESSARY OBJECTS
=================================================================================================================
'''
eos_obj = PengRobinsonEos(pC, Tc, AcF, omega_a, omega_b, kij)




'''
=================================================================================================================
CODE FOR SEEK THE BUBBLE PRESSURE
=================================================================================================================
'''

def calculate(xguess, yguess, p, T, epsilon = 1.e-10):
    x, y = xguess, yguess
    for counter in np.arange(100):
        f_L, Z_L = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(p, T, x, 'liquid')
        f_V, Z_V = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(p, T, y, 'vapor')
        PHI_L = f_L / (x * p)
        PHI_V = f_V / (y * p)
        delta = np.sum(np.abs(x * PHI_L - y * PHI_V))
        print('delta before = ', delta)
        if delta < epsilon:
            print('delta', delta)
            print('Liquid Composition', x)
            print('Vapor Composition', y)
            return x,y, counter
        else:
            x[0] = PHI_V[0] * (PHI_V[1] - PHI_L[1]) / (PHI_L[0] * PHI_V[1] - PHI_V[0] * PHI_L[1])
            x[1] = 1. - x[0]
            y[0] = x[0] * PHI_L[0] / PHI_V[0]
            y[1] = 1. - y[0]


T = 500.0
p = 2 * 1e6
xguess = np.array([0.1, 0.9])
yguess = np.array([0.9, 0.1])

x, y, counter = calculate(xguess, yguess, p, T)
print('Loop', counter)
x_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, x)
y_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, y)
print('Liquid Mass Fraction [-] = ', x_mass)
print('Vapor Mass Fraction [-] = ', y_mass)

