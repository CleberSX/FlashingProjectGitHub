import numpy as np
import Tools_Convert
from Molecule import Molecule
from kij_parameter import Kij_class
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
MM = np.array([58.12, 356.])
specific_heat_mass_base = np.array([1.663, 1.539]) * 1.e3  # [J / kg K]
specific_heat = Tools_Convert.convert_specific_heat_massbase_TO_molarbase(specific_heat_mass_base, MM)
#sort as name, molar_mass, Tc, pC, AcF, Cp
comp1 = Molecule("R600a", 58.12, (134.7 + 273.15), 36.4e5, 0.1853, specific_heat[0]) # [SOURCE II]
comp2 = Molecule("POE_ISO7", 356., (469.9 + 273.15), 11.27e5, 0.7915, specific_heat[1]) # [SOURCE II]

#sort as name, molar_mass, Tc, pC, AcF, Cp
# comp1 = Molecule("R600a", 58.12, (134.7 + 273.15), 36.4e5, 0.1853, specific_heat[0]) # [SOURCE I]
# comp2 = Molecule("POE_ISO7", 356., (796.2 + 273.15), 10.01e5, 1.1190, specific_heat[1]) # [SOURCE I]
kij = 0.01749

'''
===============================================================================================================
Do you have kij above? If you don't have it ==> uncomment the CODE's 4 lines below 
===============================================================================================================
'''
# p_C = np.array([comp1.pC, comp2.pC])
# T_C = np.array([comp1.TC, comp2.TC])
# kij_obj = Kij_class(p_C, T_C)
# kij = kij_obj.calculate_kij()
# print(kij)


if __name__ == '__main__':
    comp1.print_parameters()
    comp2.print_parameters()



def input_properties_case_artigo_moises_2013___POE_ISO_7(comp1, comp2, kij):
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n THE DATA USED IN THIS SIMULATION WAS OBTAINED FROM: ', this_function_name)
    print('\n---\n')
    '''
                                               [SOURCE I]
    A DEPARTURE-FUNCTION APPROACH TO CALCULATE THERMODYNAMIC PROPERTIES OF REFRIGERANT-OIL MIXTURES 
    
    Moisés A. Marcelino Neto, Jader R. Barbosa Jr., INTERNATIONAL JOURNAL OF REFRIGERATION 36 (2013) 972-979

    R-600a and oil AB ISO-7 {Figures 8 - 10}.
    
    
                                               [SOURCE II] 
    CARACTERIZACAO DE PROPRIEDADES TERMOFISICAS DE MISTURAS DE OLEOS LUBRIFICANTES E FLUIDOS 
    REFRIGERANTES NATURAIS, Dissertacao, (2006)
    
    Moisés A. Marcelino Neto
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


def experimental_data_case_artigo_moises_2013___POE_ISO_7():

    #T_list = np.array([10.7, 20.2, 30.2, 40.7, 50.6, 60.])
    #data from SOURCE II

    refrigerant_global_mass_fraction_expData_10Celsius = np.array([1.0, 0.7468, 0.6412, 0.6273,
                                                                    0.6021, 0.5759, 0.5235, 0.5022,
                                                                    0.462, 0.4117, 0.378, 0.2671,
                                                                    0.0773,	0.0])

    bubble_pressure_10Celsius = np.array([2.26, 1.99, 1.95, 1.93, 1.93, 1.91, 1.9,	1.88, 1.82, 1.76, 1.74,
                                           1.34, 0.56, 0.0]) * 1.e5

    refrigerant_global_mass_fraction_expData_20Celsius = np.array([1.0, 0.7098, 0.6513, 0.5892, 0.5404, 0.4775,
                                                                    0.4365, 0.3908, 0.35, 0.3063, 0.2867, 0.259,
                                                                    0.2231, 0.1903, 0.164, 0.1406, 0.0])

    bubble_pressure_20Celsius = np.array([3.04, 2.71, 2.68, 2.64, 2.6, 2.53, 2.49, 2.42, 2.35, 2.27, 2.17, 2.08,
                                           1.9, 1.81, 1.72, 1.62, 0.0]) * 1.e5

    refrigerant_global_mass_fraction_expData_30Celsius = np.array([1.0, 0.5039, 0.4454, 0.3366, 0.2746, 0.1898,
                                                                    0.1383, 0.1264, 0.0])

    bubble_pressure_30Celsius = np.array([4.06, 3.45, 3.27, 2.99, 2.68, 2.31, 2.1, 1.97, 0.0]) * 1.e5

    refrigerant_global_mass_fraction_expData_40Celsius = np.array([1.0, 0.7816, 0.6656, 0.5833, 0.486, 0.4356,
                                                                    0.349, 0.3089, 0.2332, 0.218, 0.1633,
                                                                    0.1139, 0.0])

    bubble_pressure_40Celsius = np.array([5.36, 4.99, 4.87, 4.74, 4.52, 4.39, 4.21, 4.04, 3.39, 3.19,
                                           2.69, 2.26, 0.0]) * 1.e5

    refrigerant_global_mass_fraction_expData_50Celsius = np.array([1.0, 0.8238, 0.6486, 0.497, 0.3768, 0.3254, 0.262,
                                                                    0.2179, 0.1939, 0.156, 0.1303, 0.1154, 0.0985,
                                                                    0.0866, 0.0743, 0.0637, 0.0533, 0.0])

    bubble_pressure_50Celsius = np.array([6.87, 6.68, 6.42, 5.91, 5.59, 5.22, 4.87, 4.51, 4.08, 3.68, 3.28, 2.92,
                                           2.59, 2.29, 2.05, 1.77, 1.51, 0.0]) * 1.e5

    refrigerant_global_mass_fraction_expData_60Celsius = np.array([1.0, 0.5888, 0.5239, 0.4213, 0.3624, 0.2948,
                                                                    0.2589, 0.216, 0.1873, 0.1639, 0.1171, 0.0943,
                                                                    0.0823, 0.054, 0.0])

    bubble_pressure_60Celsius = np.array([8.68, 8.37, 8.07, 7.75, 7.36, 6.94, 6.44, 5.96, 5.47, 4.97, 4.06,
                                           3.76, 3.37, 2.69, 0.0]) * 1.e5

    return (refrigerant_global_mass_fraction_expData_10Celsius, bubble_pressure_10Celsius,
            refrigerant_global_mass_fraction_expData_20Celsius, bubble_pressure_20Celsius,
            refrigerant_global_mass_fraction_expData_30Celsius, bubble_pressure_30Celsius,
            refrigerant_global_mass_fraction_expData_40Celsius, bubble_pressure_40Celsius,
            refrigerant_global_mass_fraction_expData_50Celsius, bubble_pressure_50Celsius,
            refrigerant_global_mass_fraction_expData_60Celsius, bubble_pressure_60Celsius)