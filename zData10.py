import numpy as np
import Tools_Convert
from Molecule import Molecule
import sys

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

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

comp1 = Molecule("butante", 58.122, 425.13, 3.796e6, 0.201, 1.587e3)
comp2 = Molecule("octane", 114.23, 569.32, 2.497e6, 0.395, 1.5378e3)
kij = 0.0120


if __name__ == '__main__':
    comp1.print_parameters()
    comp2.print_parameters()



def input_properties_case_REFPROP___ButaneOctane(comp1, comp2, kij):
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n THE DATA USED IN THIS SIMULATION WAS OBTAINED FROM: ', this_function_name)
    print('\n---\n')
    '''
                                                
    SOURCE: REFPROP -- BUTANE AND OCTANE.
    
    '''

    critical_pressure = np.array([comp1.pC, comp2.pC])
    critical_temperature = np.array([comp1.Tc, comp2.Tc])
    acentric_factor = np.array([comp1.AcF, comp2.AcF])
    molar_mass = np.array([comp1.MM, comp2.MM])
    omega_a = 0.45724 * np.ones_like(molar_mass)
    omega_b = 0.07780 * np.ones_like(molar_mass)


    binary_interaction = np.array(
        [[0.0000, kij],
         [kij, 0.0000]]
    )
    specific_heat_mass_base = np.array([comp1.Cp, comp2.Cp])

    specific_heat = Tools_Convert.convert_specific_heat_massbase_TO_molarbase(specific_heat_mass_base, molar_mass)

    return (critical_pressure, critical_temperature, acentric_factor, molar_mass, omega_a, omega_b,
            binary_interaction, specific_heat)


def refprop_data():

    #data from REFPROP

    refrigerant_global_mass_fraction_expData_107Celsius = np.array([1.0, 0.7468, 0.6412, 0.6273,
                                                                    0.6021, 0.5759, 0.5235, 0.5022,
                                                                    0.462, 0.4117, 0.378, 0.2671,
                                                                    0.0773,	0.0])

    bubble_pressure_107Celsius = np.array([2.26, 1.99, 1.95, 1.93, 1.93, 1.91, 1.9,	1.88, 1.82, 1.76, 1.74,
                                           1.34, 0.56, 0.0]) * 1.e5

    refrigerant_global_mass_fraction_expData_202Celsius = np.array([1.0, 0.7098, 0.6513, 0.5892, 0.5404, 0.4775,
                                                                    0.4365, 0.3908, 0.35, 0.3063, 0.2867, 0.259,
                                                                    0.2231, 0.1903, 0.164, 0.1406, 0.0])

    bubble_pressure_202Celsius = np.array([3.04, 2.71, 2.68, 2.64, 2.6, 2.53, 2.49, 2.42, 2.35, 2.27, 2.17, 2.08,
                                           1.9, 1.81, 1.72, 1.62, 0.0]) * 1.e5

    refrigerant_global_mass_fraction_expData_302Celsius = np.array([1.0, 0.5039, 0.4454, 0.3366, 0.2746, 0.1898,
                                                                    0.1383, 0.1264, 0.0])

    bubble_pressure_302Celsius = np.array([4.06, 3.45, 3.27, 2.99, 2.68, 2.31, 2.1, 1.97, 0.0]) * 1.e5

    refrigerant_global_mass_fraction_expData_406Celsius = np.array([1.0, 0.7816, 0.6656, 0.5833, 0.486, 0.4356,
                                                                    0.349, 0.3089, 0.2332, 0.218, 0.1633,
                                                                    0.1139, 0.0])

    bubble_pressure_406Celsius = np.array([5.36, 4.99, 4.87, 4.74, 4.52, 4.39, 4.21, 4.04, 3.39, 3.19,
                                           2.69, 2.26, 0.0]) * 1.e5

    refrigerant_global_mass_fraction_expData_505Celsius = np.array([1.0, 0.8238, 0.6486, 0.497, 0.3768, 0.3254, 0.262,
                                                                    0.2179, 0.1939, 0.156, 0.1303, 0.1154, 0.0985,
                                                                    0.0866, 0.0743, 0.0637, 0.0533, 0.0])

    bubble_pressure_505Celsius = np.array([6.87, 6.68, 6.42, 5.91, 5.59, 5.22, 4.87, 4.51, 4.08, 3.68, 3.28, 2.92,
                                           2.59, 2.29, 2.05, 1.77, 1.51, 0.0]) * 1.e5

    refrigerant_global_mass_fraction_expData_600Celsius = np.array([1.0, 0.5888, 0.5239, 0.4213, 0.3624, 0.2948,
                                                                    0.2589, 0.216, 0.1873, 0.1639, 0.1171, 0.0943,
                                                                    0.0823, 0.054, 0.0])

    bubble_pressure_600Celsius = np.array([8.68, 8.37, 8.07, 7.75, 7.36, 6.94, 6.44, 5.96, 5.47, 4.97, 4.06,
                                           3.76, 3.37, 2.69, 0.0]) * 1.e5

    return (refrigerant_global_mass_fraction_expData_107Celsius, bubble_pressure_107Celsius,
            refrigerant_global_mass_fraction_expData_202Celsius, bubble_pressure_202Celsius,
            refrigerant_global_mass_fraction_expData_302Celsius, bubble_pressure_302Celsius,
            refrigerant_global_mass_fraction_expData_406Celsius, bubble_pressure_406Celsius,
            refrigerant_global_mass_fraction_expData_505Celsius, bubble_pressure_505Celsius,
            refrigerant_global_mass_fraction_expData_600Celsius, bubble_pressure_600Celsius)