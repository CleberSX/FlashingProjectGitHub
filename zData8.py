import numpy as np
import Tools_Convert
from kij_parameter import Kij_class
from Molecule import Molecule
import sys

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)


'''
======================================================================================================
TO USE THIS FILE: 

(0) - FIRST ALL: change the name of the 'input_properties_case_artigo_moises_2013___POE_ISO_5()'
(1) - filled up MM, specific_heat (or specific_heat_mass).
(2) - filled up numerical values: molar_mass, Tc, pC, AcF and 'name' on vectors comp1 and comp2 
(3) - give a value to kij; if you don't have kij, you can uncomment the CODE which estimate kij
      (this parameter is built inside kij_parameter.py)     
(4) - if you have experimental data, change the function's name 
      'experimental_data_case_artigo_moises_2013___AB_ISO_5()' and update with your data 
======================================================================================================
'''


'''
=============================================================================================================
Class Molecule("name", MM, Tc, pC, AcF, Cp) 

LEGEND:
name: compost name
MM: molar mass of each component [kg/kmol]
Tc: critical temperature [K] 
pC: critical pressure [Pa]
AcF: acentric factor [-]
Cp: specific heat [J / kg K]   !<=== HERE IT IS IN MASS BASE, BUT IT IS CONVERTED TO MOLAR BASE INSIDE THE FUNCTION 
=============================================================================================================
'''


MM = np.array([58.12, 240.])
specific_heat_mass_base = np.array([1663.0, 1490.0])  #(J/kg K)
specific_heat = Tools_Convert.convert_specific_heat_massbase_TO_molarbase(specific_heat_mass_base, MM)


#sort as name, molar_mass, Tc, pC, AcF, Cp
comp1 = Molecule("R600a", 58.12, (134.7 + 273.15), 36.4e5, 0.1853, specific_heat[0])
comp2 = Molecule("AB_ISO5", 240., (675.9 + 273.15), 20.6e5, 0.9012, specific_heat[1])
kij = -0.02668

'''
===============================================================================================================
Do you have kij above? If you don't have it ==> uncomment the CODE's 4 lines below 
===============================================================================================================
'''
# p_C = np.array([comp1.pC, comp2.pC])
# T_C = np.array([comp1.TC, comp2.TC])
# kij_obj = Kij_class(p_C, T_C)
# kij = kij_obj.calculate_kij()




if __name__ == '__main__':
    comp1.print_parameters()
    comp2.print_parameters()


def input_properties_case_artigo_moises_2013___POE_ISO_5(comp1, comp2, kij):
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n THE DATA USED IN THIS SIMULATION WAS OBTAINED FROM: ', this_function_name)
    print('\n---\n')
    '''
    A DEPARTURE-FUNCTION APPROACH TO CALCULATE THERMODYNAMIC PROPERTIES OF REFRIGERANT-OIL MIXTURES 
    
    Moisés A. Marcelino Neto, Jader R. Barbosa Jr., INTERNATIONAL JOURNAL OF REFRIGERATION 36 (2013) 972-979

    R-600a and oil AB ISO-5 (alkyl benzene) {Figures 3 - 7}.
    '''
    critical_pressure = np.array([comp1.pC, comp2.pC])
    critical_temperature = np.array([comp1.TC, comp2.TC])
    acentric_factor = np.array([comp1.AcF, comp2.AcF])
    molar_mass = np.array([comp1.MM, comp2.MM])
    omega_a = 0.45724 * np.ones_like(molar_mass)
    omega_b = 0.07780 * np.ones_like(molar_mass)

    binary_interaction = np.array(
        [[0.0000, kij],
         [kij, 0.0000]]
    )
    specific_heat = np.array([comp1.Cp, comp2.Cp])

    return (critical_pressure, critical_temperature, acentric_factor, molar_mass, omega_a, omega_b,
            binary_interaction, specific_heat)


'''
Experimental data from article Moisés&Jader, A departure-function approach..., (2013)
These data were got by Carlos using a special program which read them directly from the article  
'''

def experimental_data_case_artigo_moises_2013___AB_ISO_5():
    refrigerant_global_mass_fraction_expData_23Celsius = np.array([0.0, 0.01871345, 0.047953216, 0.079532164,
                                                                   0.10994152, 0.143859649, 0.187134503,
                                                                   0.276023392, 0.319298246, 0.361403509,
                                                                   0.487719298, 0.514619883, 0.566081871,
                                                                   0.608187135,
                                                                   0.716959064, 0.815204678, 1.])

    bubble_pressure_23Celsius = np.array([0.0, 0.411764706, 0.801857585, 1.083591331, 1.408668731,
                                          1.60371517, 1.907120743, 2.123839009, 2.318885449, 2.405572755,
                                          2.708978328, 2.817337461, 2.904024768, 2.904024768, 3.099071207,
                                          3.185758514, 3.294117647]) * 1.e5

    refrigerant_global_mass_fraction_expData_40Celsius = np.array([0.0, 0.1181, 0.1673, 0.2117, 0.2643,
                                                                   0.3263, 0.4035, 0.4749, 0.5251, 0.6047, 0.6421,
                                                                   0.7099, 0.7766, 0.8585, 0.9205, 0.9883, 1.0000])
    bubble_pressure_40Celsius = np.array([0.0000, 2.0155, 2.4706, 3.0124, 3.2724, 3.7926, 4.0093, 4.0960,
                                          4.2910, 4.5294, 4.6161, 4.7895, 5.0279, 5.2012, 5.2446, 5.3096,
                                          5.3096]) * 1.e5

    refrigerant_global_mass_fraction_expData_60Celsius = np.array([0.0000, 0.0491, 0.1240, 0.1544, 0.2246,
                                                                   0.3181, 0.4667, 0.5988, 0.7088, 0.7930, 0.8456,
                                                                   0.9439, 1.0000])

    bubble_pressure_60Celsius = np.array([0.0000, 1.1252, 2.6399, 3.3972, 4.6090, 5.3663, 6.4482, 7.1190, 7.4436,
                                          7.6816, 7.8114, 8.1360, 8.6121]) * 1.e5

    refrigerant_global_mass_fraction_expData_80Celsius = np.array([0.0000, 0.0538, 0.0737, 0.1193, 0.1684, 0.2211,
                                                                   0.3158, 0.4936, 0.6760, 0.9228, 1.0000])

    bubble_pressure_80Celsius = np.array([0.0000, 1.9071, 2.4923, 3.8576, 4.9628, 6.2632, 7.8669, 9.5356, 11.2910,
                                          12.8297, 13.2632]) * 1.e5
    return (refrigerant_global_mass_fraction_expData_23Celsius, bubble_pressure_23Celsius,
            refrigerant_global_mass_fraction_expData_40Celsius, bubble_pressure_40Celsius,
            refrigerant_global_mass_fraction_expData_60Celsius, bubble_pressure_60Celsius,
            refrigerant_global_mass_fraction_expData_80Celsius, bubble_pressure_80Celsius)
