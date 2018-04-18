import numpy as np


def viscR_function(T, scale):
    '''
    This function calculate the R134a viscosity \n
    
    You must give 2 arguments when call the function: (T,'C') or (T,'K') \n
    R134a Viscosity: in [Pa.s] \n


    This correlation has been gotten from: \n 
    Tese de Jackson Braz Marcinichen (2006), page 191 \n

    "Estudo teórico e experimental da obstrução de tubos \n
    capilares por adsorção de óleo éster" \n
     '''
    
    if scale == 'K':
        Tcelsius = T - 273.15
    elif scale == 'C':
        Tcelsius = T 
    else:
        msg = 'You must specify what it\'s the temperature scale:'
        msg += ' "C" for Celsius or "K" for Kelvin'
        raise Exception(msg)

    A, B, C = 16.3254319867, -0.0789606203, -0.0001831244
    D, E = 0.0015626243, -0.00003081
    num = (A + B * Tcelsius + C * Tcelsius ** 2) 
    den = (1. + D * Tcelsius + E * Tcelsius ** 2)

    return 1e-6 * (num / den) ** 2
