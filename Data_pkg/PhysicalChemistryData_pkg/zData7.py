import numpy as np
import sys

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

def input_properties_case_noel_nevers_exF5_pg344():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n THE DATA USED IN THIS SIMULATION WAS OBTAINED FROM: ', this_function_name)
    print('\n---\n')
    '''
    PHYSICAL AND CHEMICAL EQUILIBRIUM FOR CHEMICAL ENGINEERS PROBLEM APPENDIX F (F5/10.3), PG-344 

    Ethane and n-Heptane (C2 and nC7).
    '''
    temperature = (210.0 + 459.67) * 5.0 / 9.0  # [F to K]
    pressure = 800.0 * 6894.75729  # [psi to pascal] #800

    critical_pressure = np.array([48.72, 27.4]) * 1.e5  # [bar to pascal]
    critical_temperature = np.array([305.3, 540.2])  # [K]
    acentric_factor = np.array([0.1, 0.35])  # [-]
    molar_mass = np.array([30.069, 100.203])  # [kg/kmol]
    omega_a = 0.45724 * np.ones_like(molar_mass)  # [-]
    omega_b = 0.07780 * np.ones_like(molar_mass)  # [-]

    binary_interaction = np.array(
        [[0.000000, 0.000000],
         [0.000000, 0.000000]]
    )

    global_molar_fractions = np.array([0.6, (1.0 - 0.6)])
    specific_heat = np.array([1.0, 2.0])

    return (pressure, temperature, global_molar_fractions, 
        critical_pressure, critical_temperature, acentric_factor,
        molar_mass, omega_a, omega_b, binary_interaction, specific_heat)


