
import numpy as np
from scipy.optimize import fsolve

import Tools_Convert
from EOS_PengRobinson import PengRobinsonEos
from Flash import Flash
from InputData___ReadThisFile import props  # Check this file InputData__ReadThisFile.py to see which data you are using
from Michelsen import Michelsen
from Properties import Properties
from RachfordRice import RachfordRice
from Wilson import calculate_K_values_wilson


R = 8314.  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

'''
=======================================
         WHAT DOES PROGRAM DO? - This program is used to determine the mixture's quality as well the liquid and ...
                                 ...vapor composition on a specific condition of pressure,...
                                  ...temperature and global composition (feed), that is, @(p,T,z) 
                                  

         READ THIS - parts in this file FlashAlgorithm.py you must check and change if necessary:

(i) - EXPECTED (obs.: fugacity_expected, K_value_expected and x and y_expected are useful to...
                      verify our simulation results if you have the expected values to compare)
(ii) - READING FILE DATA -  First: check what data you are reading in the InputData_ReadThisFile.py...
                                Second: the properties are imported from InputData___ReadThisFile.py. In this...
                                        ... file the places used by pressure, temperature and ...
                                        ... global_molar_fractions have been left empty
                                Third: you must specify the @(p, T, z) which you are interested (go below to set ...
                                       ... these parameters)  
(iii) - REFERENCES - it's necessary take references values if you want to calculate enthalpy, entropy...
                     ... in a specific condition 
                     (iii-a) check below which values have been used; change them if necessary
                     (iii-b) first all is necessary execute BubbleP_singlePoint.py to take pB = pB(T,z) 
                     (iii-c) the reference pressure is obtained from item (iii-b), i.e., reference_pressure = pB                      
=======================================
'''


'''
=======================================
         READING THE FILE DATA - specifying (giving names) to the variables that are inside of... 
                                 ...props list (see InputData_ReadThisFile.py). 
                                 Check what data you are reading in the InputData_ReadThisFile.py 
                               - some data places are left blank; these empty places belong to the variable we want
                                 ... to specify   
=======================================

LEGEND:
AcF: Acentric factor [-]
omega_a, omega_b: Parameters from Peng Robinson (EdE)
kij: Binary interaction factor
Cp: Specific heat in molar base [J/kmol K]
TR: Reference temperature [K]
pR: Reference pressure [Pa]
T: Temperature [K]
Tc: Critical temperature [K]
p: Pressure [Pa]
pC: Critical pressure [Pa]
props: Just a name has been given to the properties list
'''
(pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props



'''
===================================
        @(PRESSURE, TEMPERATURE, GLOBAL MOLAR FRACTION): These are the conditions you are interested to evaluate 
===================================

LEGEND:
p: Interested pressure (evaluate the enthalpy for this pressure) [Pa]
T: Interested temperature (evaluate the enthalpy considering this isotherm) [K]
OCR: oil-circulation ratio [-]: it is a mass fraction of oil per total mass mixture
z_mass: global mass fraction 
z: global molar fraction
'''
p = 0.3e5               # <=============================== change here
T = (30. + 273.15)                # <=============================== change here
LC, base = 0.40, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
z, z_mass = Tools_Convert.frac_input(MM, zin, base)



'''
=======================================
         CREATING OBJECTS - [to a better identification, all objects' names are followed by "_obj"]
=======================================
'''
rr_obj = RachfordRice()
eos_obj = PengRobinsonEos(pC, Tc, AcF, omega_a, omega_b, kij)

michelsen_obj = Michelsen()
flash_obj = Flash(max_iter = 50, tolerance = 1.0e-13, print_statistics=False)
prop_obj = Properties(pC, Tc, AcF, omega_a, omega_b, kij)



def getting_the_results_from_FlashAlgorithm_main(p, T, pC, Tc, AcF, z):
      initial_K_values = calculate_K_values_wilson(p, T, pC, Tc, AcF)
      is_stable, K_michelsen = michelsen_obj(eos_obj, p, T, z, initial_K_values, max_iter=100, tolerance=1.0e-12)
      K_flash, F_V_flash = flash_obj(rr_obj, eos_obj, p, T, z, K_michelsen)
      Vector_ToBe_Optimized = np.append(K_flash, F_V_flash)
      result = fsolve(func=flash_obj.flash_residual_function, x0=Vector_ToBe_Optimized, args=(T, p, eos_obj, z))
      size = result.shape[0]
      K_values_newton = np.array(result[0:size - 1])
      F_V = result[-1]
      return (F_V, is_stable, K_values_newton, initial_K_values)


'''
=======================================
         OTHER RESULTS (CALCULATIONS): Once with K and F_V values, the mixture's composition and other variables can be obtained
=======================================
'''
F_V, is_stable, K_values_newton, initial_K_values = getting_the_results_from_FlashAlgorithm_main(p, T, pC, Tc, AcF, z)
x = z / (F_V * (K_values_newton -1.) + 1.)
y = K_values_newton * x
f_V, Z_V = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(p, T, y, 'vapor')
f_L, Z_L = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(p, T, x, 'liquid')
rho_L = prop_obj.calculate_density_phase(p, T, MM, x, 'liquid')
vol_L = np.reciprocal(rho_L)
rho_V = prop_obj.calculate_density_phase(p, T, MM, y, 'vapor')
vol_V = np.reciprocal(rho_V)
M_V = prop_obj.calculate_weight_molar_mixture(MM, y, 'vapor')
M_L = prop_obj.calculate_weight_molar_mixture(MM, x, 'liquid')
F_V_mass = np.reciprocal( (M_L / M_V) * ( 1 / F_V - 1.) + 1.)
x_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, x)
y_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, y)




'''
=======================================
         PRINT THE RESULTS 
=======================================
'''

def func_print():
      print('System is stable?', is_stable)
      print('\n-------\nK_wilson(initial values estimates):', initial_K_values)
      print('K_values newton:', K_values_newton)
      print('\n-----\nFugacities obtained (f_L):', f_L)
      print('Fugacities obtained (f_V):', f_V)
      # print ('Fugacities expected:', fugacity_expected)

      # print ('K_values expected:', K_values_expected)
      # print ('Norm difference: %.4f' % np.linalg.norm(K_values_expected - K_values_newton))
      print('\n------\nVapor molar fraction (quality in molar base): %.4f' % F_V)
      print('\n------\nVapor mass fraction (quality in mass base): %.4f' % F_V_mass)
      print('\n------\nz_i ==> global mass fraction comp. i:', z)
      print('x_i ==> molar concentration comp. i liquid phase:', x)
      # print ('x_i ==> molar concentration expected:', x_expected)
      print('y_i ==> molar concentration comp. i vapor phase:', y)
      # print ('y_i ==> molar concentration expected:', y_expected)
      print('x_i_mass ==> mass concentration comp. i liquid phase:', x_mass)
      print('y_i_mass ==> mass concentration comp. i vapor phase:', y_mass)
      print('z_i_mass ==> global mass fraction comp. i:', z_mass)
      print('\n-------\nZ_V ==> compressibility factor comp. i vapor phase: %.2f' % Z_V)
      print('Z_L ==> compressibility factor comp. i liquid phase: %.2f' % Z_L)
      print('\n------\nM_L ==> molar weight liquid phase [kg/kmol]: %.2f' % M_L)
      print('M_V ==> molar weight vapor phase [kg/kmol]: %.2f' % M_V)
      print('\n----\nrho_L ==> liquid density [kg/m3]: %.2f' % rho_L)
      print('vol_L ==> liquid specific volume [m3/kg]: %.3e' % vol_L)
      print('\n----\nrho_V ==> vapor density [kg/m3]: %.2f' % rho_V)
      print('vol_V ==> vapor specific volume [m3/kg]: %.3e' % vol_V)


if __name__ == '__main__':
      func_print()





# THE SCRIPT BELOW WAS BUILT BEFORE THE FUNCTION getting_the_results_from_FlashAlgorithm_main() HAS BEEN CREATED.
# SO, SUCH FUNCTION REPLACES THIS ENTIRE SCRIPT
# ''
# '''
# ''''''
# ''''''''''
# '''''''''''''
# =======================================
#          STARTING THE SIMULATION - the sequence on which the methods are called
# =======================================
# '''
# # This Wilson's method estimates initial K-values: calculate the first estimation of K (ELV ratio)
# initial_K_values = calculate_K_values_wilson(p, T, pC, Tc, AcF)
#
# # This method check if the mixture is stable and gives Michelsen's K (a better estimation than Wilson's K)
# is_stable, K_michelsen = michelsen_obj(eos_obj, p, T, z, initial_K_values, max_iter=100, tolerance=1.0e-12)
#
# # Executing the Flash
# K_flash, F_V_flash = flash_obj(rr_obj, eos_obj, p, T, z, K_michelsen)
#
#
# '''
# __OPTION I__: Use estimates from Wilson's Equation!!!
# Vector_ToBe_Optimized = x0 = np.append(initial_K_values, F_V) # It does not work!
# __OPTION II__: Use estimates from Michelsen test!!!
# Vector_ToBe_Optimized = x0 = np.append(K_michelsen, F_V) # It does not work!
# __OPTION III__: Use estimates from flash (successive substitutions)!!!
# Vector_ToBe_Optimized = np.append(K_flash, F_V_flash) # Good estimate!
# '''
# Vector_ToBe_Optimized = np.append(K_flash, F_V_flash)
#
#
# '''
# Finally, optimizing my results: ask to execute this method to find the K and F_V that make the residual of
# ...fugacity and mass balance go forward to zero. This is unique objective of this method!
# '''
# result = fsolve(func=flash_obj.flash_residual_function, x0=Vector_ToBe_Optimized, args=(T, p, eos_obj, z))
#
#
#
#
# '''
# =======================================
#          EXTRACTING THE MAINLY RESULTS: K (ELV ratio) and F_V (quality: molar vapor-liquid ratio)
# =======================================
# '''
# size = result.shape[0]
#
# K_values_newton = np.array(result[0:size-1])
# F_V = result[-1]



