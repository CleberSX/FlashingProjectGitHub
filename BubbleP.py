
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
ESTA CLASS É PARA CALCULAR A PRESSAO DE BOLHA DE UMA MISTURA.
PONTOS QUE DEVEMOS SE ATENTAR:
(1) - verificar qual arquivo (dados do problema) está sendo importado no arquivo ==> InputData__ReadThisFile.py
(2) - este codigo eh para calcular um ponto de bolha para um único ponto de temperatura e concetracao
(3) - para levantar uma curva inteira de pontos bolha para varias temperaturas e varias concetracoes ==> va para...
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
y: Molar fraction of vapor phase [-]
================================================================================================================= 
'''


'''
=================================================================================================================
TO LOGGING MSG(import logging and logging.basicConfig and logging.disable). 
1st option: write on the screen
2nd option: write on the file

To use this tool you must choose uncomment one of the following options below: the 1st OR the 2nd 
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

class Bubble_class:
    def __init__(self, pC, Tc, AcF, kij):
        self.pC, self.Tc, self.AcF, self.kij = pC, Tc, AcF, kij


    def pressure_guess(self, T, z):
        '''To a specific temperature, this function provides a estimation (initial guess pressure) to find...
         ... the mixture bubble pressure. This result is used to create a range of pressure.
         Once with this pressure range, a numerical method is applied to seek the correct bubble pressure
         '''
        return np.sum((z * self.pC) * np.exp(5.37 * (1 + self.AcF) * (1 - (self.Tc / T))))


    def inner_loop(self, eos_obj, p, T, z, iteration_michelsen, tolerance_michelsen):
        '''
        This function is responsible for a more realistic K (vapor-liquid ratio) value. So, the first estimation
        of K is made with Wilson equation. After that, this K_Wilson is fed in Michelsen's code. This, on the other hand,
         notifies us if this mixture is stable. Besides that, in case of not stable phase (split in 2 phases),
         Michelsen's code also provides a better K estimation.

        :iteration_michelsen: is max iteration number used in Michelsen algorithm
        :tolerance_michelsen: is the tolerance exigence in Michelsen algorithm
        '''
        K_wilson = calculate_K_values_wilson(p, T, self.pC, self.Tc, self.AcF)
        is_stable, K = michelsen_obj(eos_obj, p, T, z, K_wilson, iteration_michelsen, tolerance_michelsen)
        return is_stable, K


    def __call__(self, T, z, iteration_michelsen = 300, tolerance_michelsen = 1.0e-20,
                      tolerance = 1.0e-20, max_loop = 1000, p_drop = 2000):
        '''
        :iteration_michelsen: is max iteration number used in Michelsen algorithm
        :tolerance_michelsen: is the tolerance exigence in Michelsen algorithm
        :tolerance: used to evaluate the fugacity ratio
        :max_loop: to avoid infinite loop in the while loop of fugacity ratio
        :p_drop: we start with a pressure 120% of that one obtained by function calculate_pressure_guess ();
                 with this procedure it is expected to warranty the fluid as compressed liquid (subcooled liquid region)
                 After that, the pressure is reduced step by step by p_drop until reach the region where ELV can
                 really exist. This procedure was implemented using Michelsen stability test.

        P.S.: we have added a handle error, in this case an 'assert' inside 'while loop'. The objective is to avoid
              getting in a infinite loop and also alerts the user to increase the loop's size.
        '''
        Sz = np.sum(z)
        assert np.abs(Sz - 1.0) < 1.e-5, 'YOUR GLOBAL MOLAR FRACTION MUST BE EQUAL = 1. Check your data' + str(Sz)
        p = 1.5 * self.pressure_guess(T, z)
        is_stable, K = self.inner_loop(eos_obj, p, T, z, iteration_michelsen, tolerance_michelsen)
        step = 0
        if is_stable:
            while is_stable:
                logging.debug('The current step is = ' + str(step))
                step += 1
                p -= p_drop
                logging.debug('Updating the Pressure =====> ' + str(p))
                is_stable, K = self.inner_loop(eos_obj, p, T, z, iteration_michelsen, tolerance_michelsen)
                logging.debug('One-phase ================================================ > ' + str(is_stable))
                assert step <= max_loop, '\n\n\t --->> If this msg pop up it\'s because you reached the max_loop. So, ' \
                                         'you must increase the current [max_loop = %i] in ' \
                                         'the {bubble_main_function}. Why? It\'s because probably you\'re still in ' \
                                         'stable region' % max_loop
        x = z #In bubble problems, the feed is totally liquid
        f_L, Z_L = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(p, T, x, 'liquid')
        PHI_L = f_L / (x * p)
        y = x * K
        f_V, Z_V = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(p, T, y, 'vapor')
        PHI_V = f_V / (y * p)
        K = PHI_L / PHI_V
        y = x * K
        Sy = np.sum(y)
        f_ratio = f_L / f_V
        counter = 0
        while (np.linalg.norm(1. - f_ratio))**2 > tolerance:
            logging.debug('The current counter is = ' + str(counter))
            counter += 1
            p *= np.sum(x * K)
            logging.debug('The pressure with K_Michelse --> ' + str(p))
            f_L, Z_L = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(p, T, x, 'liquid')
            PHI_L = f_L / (x * p)
            y = x * K
            f_V, Z_V = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(p, T, y, 'vapor')
            PHI_V = f_V / (y * p)
            K = PHI_L / PHI_V
            y = x * K
            Sy = np.sum(y)
            f_ratio = f_L / f_V
            assert counter <= max_loop, '\n\n\t --->> If this msg pop up it\'s because you reached the max_loop. So, ' \
                                     'you must increase the current [max_loop = %i] in ' \
                                     'the {bubble_main_function --> the while loop involving fugacity ratio}. ' \
                                     'Why? It\'s because probably you did not reach an ELV condition ' \
                                     'stable region' % max_loop
            logging.debug('Fugacity ratio = ' + str(f_ratio))
        return p, y, Sy, counter





'''
=========================================================================================================
  CREATING OBJECTS:
=========================================================================================================
'''
bubble_obj = Bubble_class(pC, Tc, AcF, kij)



'''
=========================================================================================================
  INPUT DATA:
=========================================================================================================
'''
T = (0. + 273.15)   # <=================================== change here
LC, base = 99./100, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
z, z_mass = Tools_Convert.frac_input(MM, zin, base)
pG = 1.2 * bubble_obj.pressure_guess(T, z)
pB, y, Sy, counter = bubble_obj(T, z)
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

