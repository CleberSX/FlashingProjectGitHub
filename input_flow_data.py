import numpy as np
import sys
from CoolProp.CoolProp import PropsSI
from InputData___ReadThisFile import props


def input_flow_data_function():
    '''
    Return: p_e, T_e, mdotL_e

    p_e: pipe's entrance pressure [Pa] \n
    T_e: pipe's entrance temperature  [K] \n 
    mdotL_e: pipe's entrance liquid mass mass ratio [kg/s] 
    '''
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name)
    print('\n---\n')

    room_pressure = 1.e5                        # -- pressao ambiente [Pa]
    temperature = (36.5 + 273.15)                         # -- temperatura de entrada [K]
    pressure = 8.92 * room_pressure             # -- pressão entrada duto [Pa]
    subcooled_liquid_mass_flow = 0.122549 #0.0076592  # -- (mass ratio subcooled liquid) [kg/s]
    #se considerar G = 609.5 kg/(m2.s) p/ D = 16mm -- mdotL = 0.122547289
    return (pressure, temperature, subcooled_liquid_mass_flow)


'''
Rodada com POE ISO puro (Tese Dalton, Tab 4.1, pg 80)

d_garganta =	0.004	m
A_garganta =	1.25664E-05	m2 (calculei)
G_POE_garganta =	609.5	kg/(m2 s) 
mdot	0.007659203	kg/s (calculei)

'''





