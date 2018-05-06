import numpy as np




def calculate_weight_molar_mixture(molar_mass, molar_fraction):
    '''
    This function calculate the molar weight of the mixture from molar weight of component "i"
    :param molar_mass: molar weight of component "i"
    '''
    MM = molar_mass
    x = molar_fraction
    return np.einsum('i,i', MM, x)


def convert_molarfrac_TO_massfrac(MM, x):
    '''
    This method convert molar fraction to mass fraction \n

    Source: Equation (1.84) - pg 32 - Transport Phenomena in Multiphase System (2006)
            Amir Faghri & Yuwen Zhang \n

    :MM:  molar weight vector of component "i" [kg/kmol], i.e., MM = ([MM1, MM2]) \n
    x: molar fraction vector [-], i.e., x = ([x1, 1.- x1]), where x2 = 1-x1 \n
    
    Mmix: molar weight of phase mixture \n 

    Return: mass fraction vector 
    '''
    
    Mmix = calculate_weight_molar_mixture(MM, x)
    x_Mmix = np.einsum('i,i->i', x, MM)
    return x_Mmix / Mmix


def convert_massfrac_TO_molarfrac(MM, fm):
    '''
    This method convert mass fraction to molar fraction (this method is very useful because
    in real world we usually measure mass fraction; however, for the calculation here, we need
    molar fraction)

    Source: Equation (1.85) - pg 32 - Transport Phenomena in Multiphase System (2006)
            Amir Faghri & Yuwen Zhang

    :MM: vector molar weight of component "i" [kg/kmol], i.e., MM = ([MM1, MM2]) \n
    fm: vector mass fraction [-], i.e., fm = ([fm1, 1-fm1]), where fm2 = 1-fm1 \n

    x = (fm/MM) / [sum(fm/MM)]

    Return: molar_fraction
    '''

    return (fm / MM) / np.einsum('i,i->', fm, 1. / MM)


def convert_specific_heat_massbase_TO_molarbase(specific_heat_mass_base, MM):
    '''
    This method convert specific heat in mass base TO molar base
    {[J / (kg K)] ===> [J / (kmol K)]}
    Cpi = Mi*Cpi_massbase

    :specific_heat_mass_base: specific heat vector in mass base [J/kgK] \n
    :MM: vector molar weight of component "i" [kg/kmol], i.e., MM = ([MM1, MM2]) \n
    '''
    Cpi_mass = specific_heat_mass_base
    return np.einsum('i,i->i', MM, Cpi_mass)



def convert_bar_TO_pascal(bar_pressure):
    return bar_pressure * 1e5


def convert_pascal_TO_bar(pascal_pressure):
    return pascal_pressure / 1e5

'''
=========================================================================================
    FUNCTION frac_input(): return the fraction in molar base (z) and mass base (z_mass)

LEGEND:
zin: it is composition you feed; can be in molar ou mass base
type: it is flag with only two options: molar or mass 
MM: molar mass each component [kg/kmol]
=========================================================================================
'''

def frac_input(MM, zin, base):
    if base is 'molar':
        z = zin
        z_mass = convert_molarfrac_TO_massfrac(MM, zin)
    else:
        assert base is 'mass', 'You must specify if it is molar ou mass' + base
        z_mass = zin
        z = convert_massfrac_TO_molarfrac(MM, zin)
    return z, z_mass