import numpy as np
import sys

def input_flow_data_function():
    '''
    p_e: pipe's entrance pressure [Pa]
    T_e: pipe's entrance temperature  [K]  
    mdotL_e: pipe's entrance liquid mass mass ratio [kg/s]
    viscG: saturated gas's dynamic viscosity [kg/(m.s)]
    viscO: lubricant oil's dynamic viscosity [kg/(m.s)]
    viscR: refrigerant's dynamic viscosity [kg/(m.s)]   
    '''
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name)
    print('\n---\n')

    room_pressure = 1.e5                    # -- pressao ambiente [Pa]
    temperature = 300.0                     # -- temperatura de entrada [K]
    pressure = 15. * room_pressure          # -- pressão entrada duto [Pa]
    viscG = 12.02e-6                        # -- (@ saturated gas) [kg/(m s)]
    viscR = 279e-6                          # -- (@ saturated liquid refrigerant's dynamic viscosity ) [kg/(m s)]
    viscO =  279e2                               # -- (@ oil's dynamic viscosity ) [kg/(m s)]
    subcooled_liquid_mass_flow = 0.2        # -- (mass ratio subcooled liquid) [kg/s]

    return (pressure, temperature, subcooled_liquid_mass_flow, viscG, viscR, viscO)