import numpy as np
import sys

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

def input_properties_case_whitson_problem_18_PR():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n THE DATA USED IN THIS SIMULATION WAS OBTAINED FROM: ', this_function_name)
    print('\n---\n')
    '''
    TEST PROBLEM PHASE BEHAVIOUR WHITSON PROBLEM 18 APPENDIX

    Methane, Butane and Decane (C1, C4 and C10).

    Properties for the Van der Waals Equation of State.

    '''
    temperature = (280.0 + 459.67) * 5.0 / 9.0  # [F to K] #(280.0 + 459.67)
    pressure = 500.0 * 6894.75729  # [psi to pascal] #500

    critical_pressure = 6894.75729 * np.array([667.8, 550.7, 304.0])  # [psia to pascal]
    critical_temperature = (5.0 / 9.0) * np.array([343.0, 765.3, 1111.8])  # [F to K]
    acentric_factor = np.array([0.011500, 0.192800, 0.490200])  # [-]
    molar_mass = np.array([16.04, 58.12, 142.29])  # [kg/kmol]
    omega_a = 0.45724 * np.ones_like(molar_mass)  # [-]
    omega_b = 0.07780 * np.ones_like(molar_mass)  # [-]

    binary_interaction = np.array(
        [[0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000],
         [0.000000, 0.000000, 0.000000]]
    )

    global_molar_fractions = np.array([0.5, 0.42, 0.08])
    specific_heat = np.array([1.0, 2.0, 3.0])

    return (pressure, temperature, global_molar_fractions, 
        critical_pressure, critical_temperature, acentric_factor,
        molar_mass, omega_a, omega_b, binary_interaction, specific_heat)

fugacity_expected = np.array([294.397, 148.342, 3.02385]) * 6894.75729
K_values_expected = np.array([6.65071, 0.890061, 0.03624])
x_expected = np.array([0.08588, 0.46349, 0.45064])
y_expected = np.array([0.57114, 0.41253, 0.01633])
