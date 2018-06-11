import numpy as np
import sys


#=======================
'What does to do in this file input_flow_data.py? ' \
'JUST uncomment one of the cases below' \
'case = (case_name, R134a_concentration, mdotLe, temperature, pressure)'

#=======================
# case = ('case0', 0.05/100, 0.122549, 36.5, 8.92)
# case = ('case1', 16.1/100, 0.122811, 36.7, 4.79)
# case = ('case2', 21.7/100, 0.122669, 36.7, 5.80)
# case = ('case3', 30.1/100, 0.122438, 36.8, 7.16)
# case = ('case4', 39.4/100, 0.122898, 36.8, 7.96)
case = ('case5', 46.7/100, 0.122897, 36.9, 7.63) # 0.122897, 36.9, 8.63
# case = ('case6', 46.7/100, 0.106608, 36.8, 8.64)
# case = ('case7', 46.7/100, 0.139066, 37.0, 8.63)
# case = ('case8', 46.7/100, 0.122780, 36.8, 8.54)
# case = ('case9', 46.7/100, 0.122924, 36.9, 8.73)
# case = ('case10', 46.7/100, 0.123271, 36.8, 8.81)



def input_flow_data_function():
    '''

    p_e: pipe's entrance pressure [Pa] \n
    T_e: pipe's entrance temperature  [K] \n 
    mdotL_e: pipe's entrance liquid mass mass ratio [kg/s] \n
    fmR134a: mass fraction of R134a in the mixture [kg R134a / kg mixture] \n

    Return: p_e, T_e, mdotL_e, fmR134a
    '''

     
    __, fmR134a, m, T, p = case
    

    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name, 'The case we are executing (-----', case[0], '-----)')
    print('\n---\n')

    room_pressure = 1.e5                        
    temperature = (T + 273.15)                         
    pressure = p * room_pressure             
    subcooled_liquid_mass_flow = m 
    mass_fraction_R134a = fmR134a
    return (pressure, temperature, subcooled_liquid_mass_flow, mass_fraction_R134a)






