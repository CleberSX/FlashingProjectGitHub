import numpy as np
import sys

def input_pipe_data_function():
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name)
    print('\n---\n')
    '''
    THESE DATA PIPE HAVE BEEN GOTTEN FROM:_______     
    '''
    diameter = (4. / 5) * 25.4 / 1000    #[m] convert {in} to {m}
    lenght = 1.0                    #[m]
    rugosity = 1.2e-3                 #[m]               

    return (diameter, lenght, rugosity)