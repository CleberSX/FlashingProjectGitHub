import numpy as np
import sys

def input_pipe_data_function():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name)
    print('\n---\n')
    '''
    THESE DATA PIPE HAVE BEEN GOTTEN FROM:_______     
    '''
    diameter = (1. ) * 25.4 / 1000    #[m]
    lenght = 1.                    #[m]
    rugosity = 1.2e-3                 #[m]               
    inclination = 70.0                 #[graus]

    return (diameter, lenght, rugosity, inclination)