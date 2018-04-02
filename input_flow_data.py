import numpy as np
import sys

def input_flow_data_function():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name)
    print('\n---\n')
    '''
    THESE DATA FLOW HAVE BEEN GOTTEN FROM:_______     
    '''
    room_pressure = 1.e5                    # -- pressao ambiente [Pa]
    temperature = 300.0                     # -- temperatura de entrada [K]
    pressure = 15. * room_pressure          # -- pressão entrada duto [Pa]
    viscG = 12.02e-6                        # -- (@ satureted gas) [kg/(m s)]
    viscF = 279e-6                          # -- (@ satureted liquid) [kg/(m s)]
    subcooled_liquid_mass_flow = 0.2        # -- (subcooled liquid) [kg/(m s)]

    return (pressure, temperature, subcooled_liquid_mass_flow, viscG, viscF)