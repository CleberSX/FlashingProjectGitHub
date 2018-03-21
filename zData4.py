import numpy as np
import sys

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

def input_properties_case_WalasEx6_6Pg326():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n THE DATA USED IN THIS SIMULATION WAS OBTAINED FROM: ', this_function_name)
    print('\n---\n')
    '''
    TEST PROBLEM STANLEY M. WALAS, Example 6.6, page 326:
    PHASE EQUILIBRIA IN CHEMICAL ENGINEERING
        
    Mixture: C3H6 + iC4H10 (propileno + isobutano)
    '''

    temperature = 344.05 #344.05 #[K]
    pressure = 20.265e5  #[pascal]

    critical_pressure = np.array([45.6, 36.0]) * 1.0e5 # [MPa]
    critical_temperature = np.array([365., 408.1]) # [K]
    acentric_factor = np.array([0.148, 0.176]) # [-]
    molar_mass = np.array([42.08, 58.123]) # [kg/kmol]
    omega_a = 0.45724 * np.ones_like(molar_mass)  # [-]
    omega_b = 0.07780 * np.ones_like(molar_mass)  # [-]

    binary_interaction = np.array(
    [[0.000,  0.00],
     [0.00,  0.000]]
    )

    global_molar_fractions = np.array([0.607, (1.0 - 0.607)])
    specific_heat = np.array([5.49, -5.146]) #J/(kg*K)

    return (pressure, temperature, global_molar_fractions, 
        critical_pressure, critical_temperature, acentric_factor,
        molar_mass, omega_a, omega_b, binary_interaction, specific_heat)



