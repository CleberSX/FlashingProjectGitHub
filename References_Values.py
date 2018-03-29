import numpy as np
import sys

def input_reference_values_function():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name)
    print('\n---\n')
    '''
  
=========================================================================================================
         REFERENCES - These references have been used in the departure functions ()
=========================================================================================================

Source: Neto, M. A. M., Barbosa, J. R. Jr, "A departure-function approach to calculate thermodynamic properties
               of refrigerant-oil mixtures", International Journal Of Refrigeration, 36, 2013, (972-979)

The references applied here follows the article above (Moisés). In this article, the SATURATED LIQUID has been adopted:  

TR = 273.15 K (0.0 Celsius)
hR_mass = 200 kJ/kg [mass base]
sR_mass = 1000 J/(kg K) [mass base]


    '''
    reference_temperature = 273.15        # [K]
    reference_enthalpy_mass_base = 200.0                    #kJ/kg [mass base]
    reference_entropy_mass_base = 1000.0                    #kJ/(kg K) [mass base]

    return (reference_temperature, reference_enthalpy_mass_base, reference_entropy_mass_base)