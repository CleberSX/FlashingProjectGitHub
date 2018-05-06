import numpy as np
import sys

def input_properties_case_ElliottEx15_8Pg598():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n THE DATA USED IN THIS SIMULATION WAS OBTAINED FROM: ',this_function_name)
    print('\n---\n')
    '''
    TEST PROBLEM PHASE BEHAVIOUR Elliott, Example 15.8, page 598: 
        
    Mixture: methanol + benzene (MeOH, Benz).   
    '''
   
    temperature = 333.35 #[K]
    pressure = 1.01325 * 1.0e5  #[bar to pascal]

    critical_pressure = np.array([8.084, 4.895]) * 1.0e6 # [MPa]
    critical_temperature = np.array([512.5, 562.05]) # [K]
    acentric_factor = np.array([0.5658, 0.2103]) # [-]
    molar_mass = np.array([32.042, 78.112]) # [kg/kmol]
    omega_a = 0.45724 * np.ones_like(molar_mass)  # [-]
    omega_b = 0.07780 * np.ones_like(molar_mass)  # [-]

    binary_interaction = np.array(
    [[0.000,  0.084],
     [0.084,  0.000]]
    )

    global_molar_fractions = np.array([0.9, (1.0 - 0.9)])
    specific_heat = np.array([5.28, 9.82]) #J/(kg*K)

    return (pressure, temperature, global_molar_fractions, 
        critical_pressure, critical_temperature, acentric_factor,
        molar_mass, omega_a, omega_b, binary_interaction, specific_heat)
    


