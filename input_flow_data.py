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
    pressure_saturation = 1. * room_pressure # pressão de saturação [Pa] @ P=P(Temperature)
    temperature = 273.15                    #[K]
    pressure = 15. * room_pressure           # pressão início duto [Pa]
    visc_g = 12.02e-6                       #kg/(m s) @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
    visc_f = 279e-6                         #kg/(m s) @ [Tsat(P_amb)] - pg 529 - Ghiaasiaan
    gas_mass_flow = 0.                      #[kg/s]
    liquid_mass_flow = 0.2                  #[kg/s]
    molar_mass = 18.                        #[kg/kmol]
    liquid_density = np.power(0.00104,-1)   #[kg/m3]

    return (pressure, pressure_saturation, temperature, gas_mass_flow, liquid_mass_flow, 
    molar_mass, visc_g, visc_f, liquid_density)