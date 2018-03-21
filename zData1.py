import numpy as np
import sys

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

def input_properties_case_AssaelEx8_1Pg230():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n THE DATA USED IN THIS SIMULATION WAS OBTAINED FROM: ', this_function_name)
    print('\n---\n')
    '''
    TEST PROBLEM Assael, Example 8.1, page 230: 
        
    Mixture: C2 + nC7    
    '''
  

    temperature = 428.71 #[K]
    pressure = 1.0e6  #[Pa]

    critical_pressure = np.array([4.871, 2.735]) * 1.0e6 # [MPa]
    critical_temperature = np.array([305.33, 540.15]) # [K]
    acentric_factor = np.array([0.099, 0.349]) # [-]
    molar_mass = np.array([30.069, 100.203]) # [kg/kmol]
    omega_a = 0.45724 * np.ones_like(molar_mass)  # [-]
    omega_b = 0.07780 * np.ones_like(molar_mass)  # [-]

    binary_interaction = np.array(
    [[0.000,  0.01],
     [0.01,  0.000]]
    )

    global_molar_fractions = np.array([0.055, 0.945])             
    specific_heat = np.array([5.49, -5.146]) #J/(kg*K)

    return (pressure, temperature, global_molar_fractions, 
        critical_pressure, critical_temperature, acentric_factor,
        molar_mass, omega_a, omega_b, binary_interaction, specific_heat)
    

