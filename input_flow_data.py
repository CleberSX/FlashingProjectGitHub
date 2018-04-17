import numpy as np
import sys
from CoolProp.CoolProp import PropsSI
from InputData___ReadThisFile import props


def input_flow_data_function():
    '''
    p_e: pipe's entrance pressure [Pa] \n
    T_e: pipe's entrance temperature  [K] \n 
    mdotL_e: pipe's entrance liquid mass mass ratio [kg/s] 
    '''
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name)
    print('\n---\n')

    room_pressure = 1.e5                        # -- pressao ambiente [Pa]
    temperature = (36.5 + 273.15)                         # -- temperatura de entrada [K]
    pressure = 8.92 * room_pressure              # -- pressão entrada duto [Pa]
    # viscG = 12.02e-6                            # -- (@ saturated gas) [Pa.s]
    # viscR = PropsSI("V", "T", temperature, "P", room_pressure,"R134a")    # -- (liquid refrigerant's dynamic viscosity ) [Pa.s]
    # viscO =  viscO_function(temperature, 'K')   # -- (@ oil's dynamic viscosity ) [Pa.s]
    subcooled_liquid_mass_flow = 0.01           # -- (mass ratio subcooled liquid) [kg/s]

    return (pressure, temperature, subcooled_liquid_mass_flow)






