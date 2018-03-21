import numpy as np
from EOS_PengRobinson import PengRobinsonEos
from Properties import Properties
from BubbleP import Bubble_class
import Tools_Convert, FlashAlgorithm_main
from InputData___ReadThisFile import props #Check this file InputData__ReadThisFile.py to see which data you are using

R = 8314.  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)


'''
=========================================================================================================
         READING THE FILE DATA - specifying (giving names) to the variables that are inside of... 
                                 ...props list (see InputData_ReadThisFile.py). 
                                 Check what data you are reading in the InputData_ReadThisFile.py 
                               - some data places are left blank; these empty places belong to the variable we want
                                 ... to specify   
=========================================================================================================
'''
(pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props


'''
=================================================================================================================
NECESSARY OBJECTS
=================================================================================================================
'''
eos_obj = PengRobinsonEos(pC, Tc, AcF, omega_a, omega_b, kij)
bubble_obj = Bubble_class(pC, Tc, AcF, kij)
prop_obj = Properties(pC, Tc, AcF, omega_a, omega_b, kij)
print(prop_obj)



'''
=========================================================================================================
         REFERENCES - These references have been used in the departure functions ()
=========================================================================================================

Source: Neto, M. A. M., Barbosa, J. R. Jr, "A departure-function approach to calculate thermodynamic properties
               of refrigerant-oil mixtures", International Journal Of Refrigeration, 36, 2013, (972-979)

The references applied here follows the article above (Moisés). In this article, the SATURATED LIQUID has been adopted:  

TR = 273.15 K (0.0 Celsius)
PR = Pb @ (TR, zi) (You MUST execute BubbleP.py to @ [TR, zi] and then copy Pb result and paste here) 
hR_mass = 200 kJ/kg [mass base]
sR_mass = 1000 J/(kg K) [mass base]


LEGEND:
AcF: acentric factor [-]
omega_a, omega_b: parameters from Peng Robinson (EdE)
kij: binary interaction factor
Cp: specific heat [J/kmol K]
TR: reference temperature [K]
pR: reference pressure [Pa] -- (You need execute the BubbleP.py, copy pB result and paste here)
T: temperature [K] (Interesting temperature)
Tc: critical temperature [K] 
p: pressure [Pa] (Interesting pressure)
pC: critical pressure [Pa]
z : global_molar_fraction (in bubble saturated condition, xi = zi)
pB: bubble pressure [Pa]
hR_mass: reference enthalpy (in the article was given in mass base, which we need transform to molar base) [J/kg]
hR: reference enthalpy (in molar base) [J/kmol]
sR_mass: reference entropy (in the article was given in mass base, which we need transform to molar base) [J/kg K]
sR: reference entropy (in molar base) [J/kmol K]
MM: molar mass of each component [kg/kmol]
M: mixture molar mass [kg/kgmol] ==> depend on phase composition 
'''



'''
=========================================================================================================
        CHANGE HERE (I)
      
        @(PRESSURE, TEMPERATURE, GLOBAL MOLAR FRACTION): These are the conditions you are interested to evaluate 
=========================================================================================================
'''
p = 200e3
T = (20. + 273.15)
LC, base = 99./100, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
z, z_mass = Tools_Convert.frac_input(MM, zin, base)



'''
=========================================================================================================
         CHANGE HERE (II)

         BUBBLE PRESSURE - First you MUST load BubbleP.py to get the bubble pressure at the interested 
                                   ... temperature. Such value is used as reference_pressure {i.e., pR = pB (T,z)}   
=========================================================================================================
'''
TR = 273.15
pR, y_sat, Sy_sat, counter = bubble_obj(TR, z)
hR_mass = 200e3
hR = hR_mass * prop_obj.calculate_weight_molar_mixture(MM, z, 'saturated_liquid')
sR_mass = 1000.
sR = sR_mass * prop_obj.calculate_weight_molar_mixture(MM, z, 'saturated_liquid')


'''
=========================================================================================================
ENTHALPY/ENTROPY CALCULATIONS: This function calculate_enthalpy_entropy() is used to check if the mixture ...
is into subcooled region or ELV. To do this, first the bubble pressure at interested temperature T is evaluated
by method bubble_obj(). It has been built this way, inside a function, to be useful when building the flow code.  
=========================================================================================================
'''

def calculate_enthalpy_entropy(p, pR, pC, T, TR, Tc, AcF, Cp, z, MM, hR, sR, printResults=False):
      pB, y_sat, Sy, counter = bubble_obj(T, z)
      if printResults:
            print('You\'re running pressure = %.3e [Pa] and temperature = %.2f C' % (p, (T - 273.15)))
            print('For this temperature the bubble pressure is: pB = %.3e [Pa] at T = (%.2f C)' % (pB, (T - 273.15)))
            print('\n---------------------------------------------------------)')
      if (p >= pB):
            F_V = 0.0
            H_subcooled = prop_obj.calculate_enthalpy(TR, T, pR, p, z, z, hR, Cp, 'liquid')
            M_L = prop_obj.calculate_weight_molar_mixture(MM, z, 'liquid')
            H_subcooled_mass = H_subcooled * np.reciprocal(M_L)
            S_subcooled = prop_obj.calculate_entropy(TR, T, pR, p, z, z, sR, Cp, 'liquid')
            S_subcooled_mass = S_subcooled * np.reciprocal(M_L)
            h, s = H_subcooled_mass, S_subcooled_mass
            if printResults:
                  print('--> This @(p,T) falls into single phase region')
                  print('\nSubcooled liquid with h = %.3e [J/ kg]' % H_subcooled_mass)
                  print('\nSubcooled liquid with s = %.3e [J/(kg K)]' % S_subcooled_mass)
      else:
            if printResults:
                  print('This @(p,T) falls into two phase region')
            F_V, is_stable, K_values_newton, initial_K_values = \
                FlashAlgorithm_main.getting_the_results_from_FlashAlgorithm_main(p, T, pC, Tc, AcF, z)
            x = z / (F_V * (K_values_newton - 1.) + 1.)
            y = K_values_newton * x
            x_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, x)
            y_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, y)
            H_sat_vapor = prop_obj.calculate_enthalpy(TR, T, pR, p, y, z, hR, Cp, 'saturated_vapor')
            M_V = prop_obj.calculate_weight_molar_mixture(MM, y, 'saturated_vapor')
            H_sat_vapor_mass = H_sat_vapor * np.reciprocal(M_V)
            H_sat_liquid = prop_obj.calculate_enthalpy(TR, T, pR, p, x, z, hR, Cp,'saturated_liquid')
            M_L = prop_obj.calculate_weight_molar_mixture(MM, x, 'saturated_liquid')
            H_sat_liquid_mass = H_sat_liquid * np.reciprocal(M_L)
            enthalpy_mixture = (1. - F_V) * H_sat_liquid + F_V * H_sat_vapor
            F_V_mass = np.reciprocal((M_L / M_V) * (1 / F_V - 1.) + 1.)
            enthalpy_mixture_mass = (1. - F_V_mass) * H_sat_liquid_mass + F_V_mass * H_sat_vapor_mass
            if printResults:
                  print('The mixture\'s state is ELV with a quality = %.3f [molar base]' % F_V)
                  print('The mixture\'s state is ELV with a quality = %.3f [mass base]' % F_V_mass)
                  print('x [molar base] = ', x)
                  print('y [molar base] = ', y)
                  print('x_mass [mass base] = ', x_mass)
                  print('y_mass [mass base] = ', y_mass)
                  print('\n======\nThe mixture liquid/vapor with h = %.3e [J/ kmol]' % enthalpy_mixture)
                  print('The mixture liquid/vapor with h_mass = %.3e [J/ kg] {mass base}' % enthalpy_mixture_mass)
            S_sat_vapor = prop_obj.calculate_entropy(TR, T, pR, p, y, z, sR, Cp, 'vapor')
            S_sat_vapor_mass = S_sat_vapor * np.reciprocal(M_V)
            S_sat_liquid = prop_obj.calculate_entropy(TR, T, pR, p, x, z, sR, Cp, 'liquid')
            S_sat_liquid_mass = S_sat_liquid * np.reciprocal(M_L)
            entropy_mixture = (1. - F_V) * S_sat_liquid + F_V * S_sat_vapor
            entropy_mixture_mass = (1. - F_V_mass) * S_sat_liquid_mass + F_V_mass * S_sat_vapor_mass
            h, s = enthalpy_mixture_mass, entropy_mixture_mass
            if printResults:
                  print('\n======\nThe mixture liquid/vapor with s = %.3e [J/(kmol K)]' % entropy_mixture)
                  print('The mixture liquid/vapor with s_mass = %.3e [J/(kg K)] {mass base}' % entropy_mixture_mass)
      return F_V, h, s





'''
=========================================================================================================
executing ...  calculate_enthalpy_entropy(p, pR, pC, T, TR, Tc, AcF, Cp, z, MM, hR, sR, printResults=False)
=========================================================================================================
'''

F_V, h, s = calculate_enthalpy_entropy(p, pR, pC, T, TR, Tc, AcF, Cp, z, MM, hR, sR, True)

