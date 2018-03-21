import numpy as np




def calculate_weight_molar_mixture(molar_mass, molar_fraction):
    '''
    This function calculate the molar weight of the mixture from molar weight of component "i"
    :param molar_mass: molar weight of component "i"
    '''
    MM = molar_mass
    x = molar_fraction
    return np.einsum('i,i', MM, x)


def convert_molarfrac_TO_massfrac(molar_mass, molar_fraction):
    '''
    This method convert molar fraction to mass fraction

    Source: Equation (1.84) - pg 32 - Transport Phenomena in Multiphase System (2006)
            Amir Faghri & Yuwen Zhang

    :molar_mass: molar weight of component "i"
    :fluid_type: just set "liquid' or 'vapor'
    Mmix: molar weight of phase mixture
    '''
    MM = molar_mass
    x = molar_fraction
    Mmix = calculate_weight_molar_mixture(MM, x)
    x_Mmix = np.einsum('i,i->i', x, MM)
    return x_Mmix / Mmix


def convert_massfrac_TO_molarfrac(molar_mass, mass_fraction):
    '''
    This method convert mass fraction to molar fraction (this method is very useful because
    in real world we usually measure mass fraction; however, for the calculation here, we need
    molar fraction)

    Source: Equation (1.85) - pg 32 - Transport Phenomena in Multiphase System (2006)
            Amir Faghri & Yuwen Zhang

    :molar_mass: molar weight of component "i"


    x = (fm/MM) / [sum(fm/MM)]
    '''
    MM = molar_mass
    fm = mass_fraction
    return (fm / MM) / np.einsum('i,i->', fm, 1. / MM)


def convert_specific_heat_massbase_TO_molarbase(specific_heat_mass_base, molar_mass):
    '''
    This method convert specific heat in mass base TO molar base
    {[J / (kg K)] ===> [J / (kmol K)]}
    Cpi = Mi*Cpi_massbase

    :specific_heat_mass_base: specific heat in mass base [J / kg K] of component "i"
    :molar_mass: molar weight of component "i"
    '''
    Cpi_mass = specific_heat_mass_base
    MM = molar_mass
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