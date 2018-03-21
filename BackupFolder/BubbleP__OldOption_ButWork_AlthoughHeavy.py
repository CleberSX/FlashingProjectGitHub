
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
# logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
#2nd OPTION
# logging.basicConfig(filename='Cleber_File.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#logging.disable(logging.CRITICAL)


'''
=================================================================================================================
PROPS IS A LIST OF PROPERTIES; BLANK SPACE CORRESPOND TO THE PROPERTY WE ARE INTERESTED. 

CHECK BubbleP__singlePoint.py OR BubbleP__curvePoints.py to see how the empty spaces are filled
=================================================================================================================
'''
(pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props



'''
=================================================================================================================
NECESSARY OBJECTS
=================================================================================================================
'''
eos_obj = PengRobinsonEos(pC, Tc, AcF, omega_a, omega_b, kij)
michelsen_obj = Michelsen()



'''
=================================================================================================================
CODE FOR SEEK THE BUBBLE PRESSURE
=================================================================================================================
'''

def calculate_pressure_guess(z, T, pC, Tc, AcF):
    '''To a specific temperature, this function provides a estimation (initial guess pressure) to find...
     ... the mixture bubble pressure. It's result is used to create a range of pressure.
     Once with this pressure range, a numerical method is applied to seek the correct bubble pressure
     '''
    return np.sum((z * pC) * np.exp(5.37 * (1 + AcF) * (1 - (Tc / T))))


def function_inner_loop(p, T, z, max_iter = 20, tolerance = 1.0e-20, stability=False):
    '''
    These function_inner_loop and function_outer_loop are functions based on Elliott's BUBBLE P algorithm (pg 595)

    INTRODUCTORY CHEMICAL ENGINEERING THERMODYNAMICS, 2nd edition, 2012
    J. RICHARD ELLIOTT and CARL T. LIRA
    '''
    logging.debug('Updating the Pressure =====> ' + str(p))
    xi = z
    Sz = np.sum(z)
    assert np.abs(Sz - 1.0) < 1.e-5, 'YOUR GLOBAL MOLAR FRACTION MUST BE EQUAL = 1. Check your data' + str(Sz)
    initial_K_values = calculate_K_values_wilson(p, T, pC, Tc, AcF)
    is_stable, K = michelsen_obj(eos_obj, p, T, z, initial_K_values, max_iter, tolerance)
    if is_stable==True:
        Sy = 1e10
        yi = np.zeros_like(z)
    else:
        if stability:
            if not is_stable:
                msg = str('===> two-phases can be in equilibrium')
            else:
                msg = str('===> one phase in this @')
            print(str(msg))
        logging.debug('K_wilson = ' + str(initial_K_values))
        logging.debug('K_michelsen = ' + str(K))
        logging.debug('One-phase ================================================ > ' + str(is_stable))
        yi = xi * K
        f_L, Z_L = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(p, T, xi, 'liquid')
        Sy = np.sum(yi)
        Sy0 = 2 * Sy
        max_loop = 2
        for counter in np.arange(max_loop):
            f_V, Z_V = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(p, T, yi, 'vapor')
            np.seterr(divide='ignore', invalid='ignore') #Because we can have PHI_V near zero for the no volatile component
            K *= f_L / f_V
            yi = xi * K
            Sy = np.sum(yi)
            logging.debug('iteracao' + str(counter))
            logging.debug('K' + str(K))
            logging.debug('yi' + str(yi))
            logging.debug('Sy = ' + str(Sy) + '\n----')
            if (np.abs(Sy - Sy0) < tolerance):
                break
            else:
                yi /= Sy
                Sy0 = np.copy(Sy)
    return Sy, yi

def function_outer_loop(p, T, z):
    '''
        These function_inner_loop and function_outer_loop are functions based on Elliott's BUBBLE P algorithm (pg 595)

        INTRODUCTORY CHEMICAL ENGINEERING THERMODYNAMICS, 2nd edition, 2012
        J. RICHARD ELLIOTT and CARL T. LIRA
        '''
    Sy, yi = function_inner_loop(p, T, z)
    logging.debug('Pressure ========> ' + str(p))
    logging.debug('Valor de (Sy - 1.0) = ' + str(Sy -1.))
    return np.abs(Sy - 1.0)


def executable(step, pL, pH, T, z):
    '''
    This function applies a numeric method to solve function_outer_loop when a range of pressure
    is provided. These pressure range is built with calculate_pressure_guess() function
    '''
    #=======================
    # ATTENTION: when brentq is applied, the return of the function_outer_loop() must NOT be absolute value
    # pB = brentq(function_outer_loop, pL, pH, args=(T, z))
    # =======================
    # ATTENTION: when fsolve is applied, the return of the function_outer_loop() must NOT be absolute value
    # x0 = (pL + pH) / 2.
    # pB = fsolve(function_outer_loop, x0, args=(T, z))
    # =======================
    ranges = (slice(pL, pH, step),)  # Go from Start to End applying this step size...
    #                                                      #... (step length instead quantity of points)
    #ATTENTION: when brute is applied, the return of the function_outer_loop() must be absolute value
    pB = brute(function_outer_loop, ranges, args=(T, z), finish=None)
    # =======================
    return pB


'''
=========================================================================================================
  INPUT DATA:
=========================================================================================================
'''
T = (20. + 273.15)   # <=================================== change here
LC, base = 0.10, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
z, z_mass = Tools_Convert.frac_input(MM, zin, base)
pG = calculate_pressure_guess(z, T, pC, Tc, AcF)
pL, pH = 0.7 * pG, 1.3 * pG
step = 5
pB = executable(step, pL, pH, T, z)
Sy, y = function_inner_loop(pB, T, z)
y_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, y)




'''
=========================================================================================================
  PRINT RESULTS:
=========================================================================================================
'''
if __name__== '__main__':
    print('\n---------------------------------------------------')
    print('[1] - Guess pB [Pa]= %.8e' % pG)
    print('[2] - ======> at T = %.2f [C], pB = %.8e [Pa] ' % ((T - 273.15), pB) + '\n')
    print('[3] - Concentration vapor phase [molar] = ', y.round(3))
    print('[4] - Concentration vapor phase [mass] = ', y_mass.round(3))
    print('[5] - Pay attention if Sy is close to unity (Sy = %.10f) [molar]' % Sy)
    print('[6] - Global {mass} fraction = ', z_mass.round(3))
    print('[7] - Global {molar} fraction = ', z.round(3))