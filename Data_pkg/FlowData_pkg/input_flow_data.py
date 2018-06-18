import numpy as np
import sys


#=======================
'What does to do in this file input_flow_data.py? ' \

'JUST uncomment one of the following cases'
#=======================
# data = {'case':'case0', 'mass_fraction_R134a': 0.15/100, 'mass_flow_rate': 0.122549, 'temperature': 36.5, 'pressure': 8.92}
# data = {'case':'case1', 'mass_fraction_R134a': 16.1/100, 'mass_flow_rate': 0.122811, 'temperature': 36.7, 'pressure': 4.79}
# data = {'case':'case2', 'mass_fraction_R134a': 21.7/100, 'mass_flow_rate': 0.122669, 'temperature': 36.7, 'pressure': 5.80}
# data = {'case':'case3', 'mass_fraction_R134a': 39.4/100, 'mass_flow_rate': 0.122438, 'temperature': 36.8, 'pressure': 7.16} #39.4/100, 36.8 graus 7.16 bar
data = {'case':'case4', 'mass_fraction_R134a': 46.7/100, 'mass_flow_rate': 0.122898, 'temperature': 36.8, 'pressure': 7.3} #7.96
# data = {'case':'case5', 'mass_fraction_R134a': 46.7/100, 'mass_flow_rate': 0.122897, 'temperature': 36.9, 'pressure': 8.63}
# data = {'case':'case6', 'mass_fraction_R134a': 46.7/100, 'mass_flow_rate': 0.106608, 'temperature': 36.8, 'pressure': 8.64}
# data = {'case':'case7', 'mass_fraction_R134a': 46.7/100, 'mass_flow_rate': 0.139066, 'temperature': 37.0, 'pressure': 8.63}
# data = {'case':'case8', 'mass_fraction_R134a': 46.7/100, 'mass_flow_rate': 0.122780, 'temperature': 36.8, 'pressure': 8.54}
# data = {'case':'case9', 'mass_fraction_R134a': 46.7/100, 'mass_flow_rate': 0.122924, 'temperature': 36.9, 'pressure': 8.73}
# data = {'case':'case10', 'mass_fraction_R134a': 46.7/100, 'mass_flow_rate': 0.123271, 'temperature': 36.8, 'pressure': 8.81}



def input_flow_data_function():
    '''

    This function was created in a separated python file to read the experimental data \n

    Return: entrance pipe's variables
    '''

    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name, 'The case we are executing (-----', data['case'], '-----)')
    print('\n---\n')

    room_pressure = 1.e5
    temperature = data['temperature'] + 273.15
    pressure = data['pressure'] * room_pressure
    subcooled_liquid_mass_flow = data['mass_flow_rate']
    mass_fraction_R134a = data['mass_fraction_R134a']

    return pressure, temperature, subcooled_liquid_mass_flow, mass_fraction_R134a






