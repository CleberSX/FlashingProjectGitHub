import numpy as np
import sys

def input_pipe_data_function():
    '''
    THESE DATA PIPE HAVE BEEN GOTTEN FROM: \n
    Tese de Doutorado de Dalton Bertoldi, (2014) \n
    
    Investigação Experimental de Escoamentos Bifásicos \n
    com Mudança de Fase de Uma Mistura Binária em um Tubo de Venturi, (2014) \n
    ======================================================================== \n

    angleVenturi_in: entrance venturi angle [rad] \n
    angleVenturi_out: outlet venturi angle [rad] \n
    diameter (D): pipe diameter (except venturi) [m] \n
    lenght (L): total pipe lenght [m] \n
    rugosity (ks): pipe rugosity [m] \n
    diameterVenturiThroat (Dvt): venturi throat diameter [m] \n
    initialVenturiCoordinate (liv): coordinate where venturi begins [m] \n
    initialVenturiThroatCoordinate (lig): coordinate where venturi throat begins [m] \n
    lastVenturiThroatCoordinate (lfg): coordinate where venturi throat ends [m] \n
    lastVenturiCoordinate (lfv): coordinate where venturi ends [m]
    '''
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name)
    print('\n---\n')

    angleVenturi_in = np.deg2rad(39.96621 / 2.)                                   
    angleVenturi_out = np.deg2rad(13.99038 / 2.)                                   
    rugosity = 2.60e-3 #1.5e-3 , 2.6e-3                      # ks
    lenght = 1100e-3                                        # L
    diameter = 16e-3                                        # D
    diameterVenturiThroat = 4e-3                            # Dvt
    initialVenturiCoordinate = 682e-3                       # ziv
    initialVenturiThroatCoordinate = 698.5e-3               # zig
    lastVenturiThroatCoordinate = 701.5e-3                  # zfg
    lastVenturiCoordinate = 750.4e-3                        # zfv


    return (angleVenturi_in, angleVenturi_out, rugosity, lenght, diameter, diameterVenturiThroat,
            initialVenturiCoordinate, initialVenturiThroatCoordinate, 
            lastVenturiThroatCoordinate, lastVenturiCoordinate)




(angleVenturi_in, angleVenturi_out, ks, L, D, Dvt, liv, lig, lfg, lfv) = input_pipe_data_function()


def areaVenturiPipe_function(l):
    ''' 
    THIS FUNCTION CALCULATES THE PIPE CROSS SECTION AREA, WHICH DEPENDS ON 
    DUCT POSITION, i.e., Ac = Ac(l)  \n

    

    angleVenturi_in: entrance venturi angle [rad] \n
    angleVenturi_out: outlet venturi angle [rad] \n
    D: pipe diameter [m] \n
    Dvt: venturi throat diameter [m] \n
    liv: coordinate where venturi begins [m] \n
    lig: coordinate where venturi throat begins [m] \n
    lfg: coordinate where venturi throat ends [m] \n
    lfv: coordinate where venturi ends [m]
    l: any pipe position [m] \n
    rc: cross section radius, i.e., rc = rc(l) [m] \n
    Ac: area cross section where Ac = Ac(l) [m2] \n

    Return: Ac
    '''
    
    #global variables (angleVenturi_in, angleVenturi_out, Ld, D, Dvt, ziv, zig, zfg, zfv)


    rvt = Dvt / 2.
    if (liv < l < lig): rc = rvt + (lig - l) * np.tan(angleVenturi_in)
    elif (lig <= l <= lfg): rc = rvt
    elif (lfg < l < lfv): rc = rvt + (l - lfg) * np.tan(angleVenturi_out)
    else: rc = D / 2.   
    msg = '\n==========================================================================\n'
    msg += 'If this msg pop up it\'s because the venturi\'s radius = %s was calculate \n'
    msg += ' as been bigger than tube\'s one = %s in the areaVenturiPipe_function(). \n'
    msg += ' So, you must review the input geometric venturi data'
    msg += '\n==========================================================================\n'
    assert rc <= D / 2,  msg  % (str(rc), str(D / 2))

    Ac = np.pi * rc ** 2 
    
    return Ac





    

