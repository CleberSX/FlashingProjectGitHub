import numpy as np
import sys

def input_flow_data_function():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name)
    print('\n---\n')
    '''
    THESE DATA FLOW HAVE BEEN GOTTEN FROM:_______     
    '''
    room_pressure = 1.e5        # pressao ambiente [Pa]
    temperature = 300.0                    #[K]
    pressure = 15. * room_pressure           # pressão início duto [Pa]
    visc_g = 12.02e-6                       #kg/(m s) @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
    visc_f = 279e-6                         #kg/(m s) @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
    gas_mass_flow = 0.                      #[kg/s]
    liquid_mass_flow = 0.2                  #[kg/s]

    return (pressure, temperature, gas_mass_flow, liquid_mass_flow, visc_g, visc_f)