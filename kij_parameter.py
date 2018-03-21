R = 8314  # [J/(kmol*K)]

'''
======================================================================================================
TO USE THIS FILE: 

(1) - import the file InputData___ReadThisFile and, inside it, read the pC and TC for binary!.
(2) - after that, just create the object like as the example below, where you must give pC and TC (just this!)
======================================================================================================
'''


class Kij_class:
    def __init__(self, pC, TC):
        self.pC, self.TC = pC, TC

    def critical_volume(self):
        ''':Source: Applied Hydrocarbon Thermodynamics - Volume 1
            Practical Thermodynamics tools for solving process engineering problems
            Wayne C. Edmister and Byung Ik Lee
            This function calculate_departure_enthalpy was based on equation (5.85A) - Edmister
            Equation (5.31) - Peng Robinson EoS
            '''
        return ( 0.307 * R * self.TC / self.pC )

#Vc = np.array([(critical_volume(comp1.pC, comp1.TC)), (critical_volume(comp2.pC, comp2.TC))])

    def calculate_kij(self):
        '''
        Source: Thermodynamics, Sanford Klein; Gregory Nellis
        Equation (11-80) - pg 741 - was developed by Poling et al. (see Source)'''
        Vc = self.critical_volume()
        arg = 2 * ( (Vc[0] * Vc[1])**(1./6) ) / (Vc[0]**(1./3) + Vc[1]**(1./3))
        return ( 1. - (arg)**3 )



