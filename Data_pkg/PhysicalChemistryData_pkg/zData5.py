import numpy as np
import sys

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

def input_properties_case_SandlerPg500():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n THE DATA USED IN THIS SIMULATION WAS OBTAINED FROM: ', this_function_name)
    print('\n---\n')
    '''
    TEST PROBLEM Sandler, Illustration 10.1-1, page 500: 
        
    Mixture: nC5 + nC7    
    '''
  

    temperature = 325.0 #[K]
    pressure = 1.013 * 1.0e5  #[bar to pascal]

    critical_pressure = np.array([3.375, 2.735]) * 1.0e6 # [MPa]
    critical_temperature = np.array([469.8, 540.15]) # [K]
    acentric_factor = np.array([0.251, 0.349]) # [-]
    molar_mass = np.array([72.150, 100.203]) # [kg/kmol]
    omega_a = 0.45724 * np.ones_like(molar_mass)  # [-]
    omega_b = 0.07780 * np.ones_like(molar_mass)  # [-]

    binary_interaction = np.array(
    [[0.000,  0.00],
     [0.00,  0.000]]
    )

    global_molar_fractions = np.array([0.3, 0.7])
    specific_heat = np.array([5.49, -5.146]) #J/(kg*K)

    return (pressure, temperature, global_molar_fractions, 
        critical_pressure, critical_temperature, acentric_factor,
        molar_mass, omega_a, omega_b, binary_interaction, specific_heat)
    

