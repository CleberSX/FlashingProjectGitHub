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
    lenght (Ld): total pipe lenght [m] \n
    rugosity (ks): pipe rugosity [m] \n
    diameterVenturiThroat (Dvt): venturi throat diameter [m] \n
    initialVenturiCoordinate (ziv): coordinate where venturi begins [m] \n
    initialVenturiThroatCoordinate (zig): coordinate where venturi throat begins [m] \n
    lastVenturiThroatCoordinate (zfg): coordinate where venturi throat ends [m] \n
    lastVenturiCoordinate (zfv): coordinate where venturi ends [m] 
    '''
    this_function_name = sys._getframe().f_code.co_name
    print('\n---\n DATA FLOW ', this_function_name)
    print('\n---\n')

    angleVenturi_in = np.deg2rad(40. / 2)                                   
    angleVenturi_out = np.deg2rad(14. / 2)                                   
    rugosity = 1.2e-3                                       # ks
    lenght = 1100e-3                                         # Ld
    diameter = 16e-3                                        # D
    diameterVenturiThroat = 4e-3                            # Dvt
    initialVenturiCoordinate = 682e-3                       # ziv
    initialVenturiThroatCoordinate = 698.5e-3               # zig
    lastVenturiThroatCoordinate = 701.5e-3                  # zfg
    lastVenturiCoordinate = 750.4e-3                          # zfv


    return (angleVenturi_in, angleVenturi_out, rugosity, lenght, diameter, diameterVenturiThroat,
            initialVenturiCoordinate, initialVenturiThroatCoordinate, 
            lastVenturiThroatCoordinate, lastVenturiCoordinate)




def areaVenturiPipe_function(angleVenturi_in, angleVenturi_out, D, Dvt, ziv, zig, zfg, zfv, z):
    ''' 
    THIS FUNCTION CALCULATES THE PIPE CROSS SECTION AREA, WHICH DEPENDS ON 
    AT DUCT POSITION, i.e., Ac = Ac(z)  \n

    angleVenturi_in: entrance venturi angle [rad] \n
    angleVenturi_out: outlet venturi angle [rad] \n
    D: pipe diameter [m] \n
    Dvt: venturi throat diameter [m] \n
    ziv: coordinate where venturi begins [m] \n
    zig: coordinate where venturi throat begins [m] \n
    zfg: coordinate where venturi throat ends [m] \n
    zfv: coordinate where venturi ends [m] 
    z: pipe position [m] \n 
    rc: cross section radius, i.e., rc = rc(z) [m] (valid for the intire circuit, including the venturi)
    '''


    if (ziv < z < zig): rc = (Dvt / 2) + (zig - z) * np.tan(angleVenturi_in)
    elif (zig <= z <= zfg): rc = Dvt / 2 
    elif (zfg < z < zfv): rc = (Dvt / 2) + (z-zfg) * np.tan(angleVenturi_out)
    else: rc = D / 2 

    #to make sure the pipe diameter is the biggest
    #if rc > (D / 2): rc = (D / 2)

    return np.pi * rc ** 2 
        



    

