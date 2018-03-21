import numpy as np
import sys

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

def input_properties_case_AssaelEx8_2Pg231():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n THE DATA USED IN THIS SIMULATION WAS OBTAINED FROM: ', this_function_name)
    print('\n---\n')
    '''
    TEST PROBLEM Assael, Example 8.2, page 231: 
        
    Mixture: CO2 + C2    
    '''
  

    temperature = 263.15 #[K]
    pressure = 3.025e6  #[pascal]

    critical_pressure = np.array([7.380, 4.871]) * 1.0e6 # [MPa]
    critical_temperature = np.array([304.1, 305.33]) # [K]
    acentric_factor = np.array([0.239, 0.099]) # [-]
    molar_mass = np.array([44.010, 30.069]) # [kg/kmol]
    omega_a = 0.45724 * np.ones_like(molar_mass)  # [-]
    omega_b = 0.07780 * np.ones_like(molar_mass)  # [-]

    binary_interaction = np.array(
    [[0.000,  0.124],
     [0.124,  0.000]]
    )

    global_molar_fractions = np.array([0.69, 0.31])             
    specific_heat = np.array([5.49, -5.146]) #J/(kg*K)

    return (pressure, temperature, global_molar_fractions, 
        critical_pressure, critical_temperature, acentric_factor,
        molar_mass, omega_a, omega_b, binary_interaction, specific_heat)

