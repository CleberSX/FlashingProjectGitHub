import numpy as np
import sys
from CoolProp.CoolProp import PropsSI
from Thermo_pkg.ThermoTools_pkg import Tools_Convert
from Data_pkg.PhysicalChemistryData_pkg.Molecule import Molecule

from Thermo_pkg.ThermoTools_pkg.kij_parameter import Kij_class


R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)



'''
======================================================================================================
TO USE THIS FILE: 

(0) - FIRST ALL: change the name of the 'input_properties_case_artigo_moises_2013___POE_ISO_5() in the function below'
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

'''OBS-1: For while, I don't have specific heat lubrificant oil POE-ISO-VG-10'''

MM = np.array([102.03, 425.]) #425.
specific_heat_mass_base = np.array([0.82421, 2.45681]) * 1.e3  # [J / kg K] @ 20ºC (Coolpropgit available on: http://ibell.pythonanywhere.com; see OBS-1)
specific_heat = Tools_Convert.convert_specific_heat_massbase_TO_molarbase(specific_heat_mass_base, MM)

#sort as name, molar_mass, Tc, pC, AcF, Cp
comp1 = Molecule("R134a", MM[0], 374.21, 40.5928e5, 0.32684, specific_heat[0])            # [SOURCE II]
comp2 = Molecule("POE-ISO-10", MM[1], (595.8 + 273.15), 6.92e5, 1.0659, specific_heat[1]) # [SOURCE I] ¿specific_heat POE ISO VG 10 não tem no artigo? 
kij = -0.002851                                                                           # [SOURCE I]




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



def input_properties_case_R134a___POE_ISO_10(comp1, comp2, kij):
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n THE DATA USED IN THIS SIMULATION WAS OBTAINED FROM: ', this_function_name)
    print('\n---\n')
    '''
                                               [SOURCE I]
    CONVECTION-DRIVEN ABSORPTION OF R-1234yf IN LUBRIFICATING OIL
    
    Moisés A. Marcelino Neto, Jader R. Barbosa Jr., INTERNATIONAL JOURNAL OF REFRIGERATION 44 (2014) 151-160

    POE ISO VG 10 {Pages 155 - 156}.
    
    
                                               [SOURCE II] 
    Coolpropgit available on: http://ibell.pythonanywhere.com 
    R-134a
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


