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



def jpDias_liquidPhaseDensity(T, p, xR_mass):
    ''' 
    The correlation was copied from JP Dias's Thesis (pg 294, EQ A.2) \n

    ESCOAMENTO DE ÓLEO E REFRIGERANTE PELA FOLGA PISTÃO-CILINDRO DE 
    COMPRESSORES HERMÉTICOS ALTERNATIVOS (2012) - UFSC \n

    T: temperature [K] \n
    p: pressure [Pa] \n
    xR_mass: vector mass concentration [-] (xR_mass = ([xR_mass, xO_mass])) \n

    This correlation is valid only in interval 20C < temp. celsius < 120C \n

    Return: densL
     '''
    wr = xR_mass[0] * 100.
    Tc = T - 273.15

    densO = 966.43636 - 0.57391608 * Tc - 0.00024475524 * Tc ** 2
    densR = PropsSI("D", "T", T, "P", p,"R134a")
    densL = densO * np.power( (1. + wr * (densO / densR - 1.) ), -1)
    return densL


def jpDias_liquidViscosity(T, p, xR_mass):

    ''' 
    The correlation was copied from JP Dias's Thesis (pg 294, EQ A.4) \n

    ESCOAMENTO DE ÓLEO E REFRIGERANTE PELA FOLGA PISTÃO-CILINDRO DE 
    COMPRESSORES HERMÉTICOS ALTERNATIVOS (2012) - UFSC \n

    T: temperature [K] \n
    p: pressure [Pa] \n
    xR_mass: vector mass concentration [-] (xR_mass = ([xR_mass, xO_mass])) \n

    This correlation is valid only in interval: \n
     0 < temp. celsius < 120C and 0.0 < refrigerant < 50.0% \n

    Return: viscL [Pa.s]
     '''

    Tc = T - 273.15
    wr = xR_mass[0] * 100

    (a1, a2) = (38.31853120, 1.0)
    (b1, b2) = (0.03581164, 0.05188487)
    (c1, c2) = (- 0.55465145, 0.02747679)
    (d1, d2) = (- 6.02449153e-5, 9.61400978e-4)
    (e1, e2) = (7.67717272e-4, 4.40945724e-4)
    (f1, f2) = (-2.82836964e-4, 1.10699073e-3)

    num = ( a1 + b1 * Tc + c1 * wr + d1 * np.power(Tc, 2) +
          e1 * np.power(wr, 2) + f1 * Tc * wr) 

    den = ( a2 + b2 * Tc + c2 * wr + d2 * np.power(Tc, 2) +
          e2 * np.power(wr, 2) + f2 * Tc * wr )

    viscCinem = num / den
    densL = jpDias_liquidPhaseDensity(T, p, xR_mass)
    return viscCinem * densL * 1e-6
     

def jpDias_liquidSpecificHeat(T, p, xR_mass):
    ''' 
    The correlation was copied from JP Dias's Thesis (pg 295, EQ A.7) \n

    ESCOAMENTO DE ÓLEO E REFRIGERANTE PELA FOLGA PISTÃO-CILINDRO DE 
    COMPRESSORES HERMÉTICOS ALTERNATIVOS (2012) - UFSC \n

    T: temperature [K] \n
    p: pressure [Pa] \n
    xR_mass: vector mass concentration [-] (xR_mass = ([xR_mass, xO_mass])) \n

    This correlation is valid only in interval: ? \n
     

    Return: cpL [J/kg K] (?...tenho verificar se são essas as unidades!!)
     '''
    Tc = T - 273.15
    wr = xR_mass[0] 
    
    cpR = PropsSI("Cpmass", "T", T, "P", p,"R134a")
    cpO = 2411.5968 + 2.260872 * Tc
    return (1. - wr) * cpO + wr * cpR



T = 36.5 + 273.15
p = 8.92e5
LC = 0.991/100
xR_mass = ([LC, (1.-LC)]) 

viscL_usandojp = jpDias_liquidViscosity(T, p, xR_mass)
viscR134a = PropsSI("V", "T", T, "P", p,"R134a")

print('densidade jpDias', viscL_usandojp)
print('densidade R134a', viscR134a)
