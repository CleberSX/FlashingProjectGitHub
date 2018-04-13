import numpy as np 

R = 8314.34                         # J / (kmol K)


class FlowTools_class():
    '''THIS CLASS IS NECESSARY TO CALCULATE FLUID PROPERTIES: SINGLE PHASE & TWO PHASE \t
    viscG: saturated gas's dynamic viscosity [kg/(m.s)] \t
    viscO: lubricant oil's dynamic viscosity [kg/(m.s)] \t
    viscR: refrigerant's dynamic viscosity [kg/(m.s)] \t
    D: diameter [m] \t
    Gt: superficial total mass flux [(kg/s)/m2]
    '''
    def __init__(self, viscG, viscR, viscO, D, Gt):
        self.viscG, self.viscR, self.viscO, self.D, self.Gt = viscG, viscR, viscO, D, Gt
    
    
    def __str__(self):
        msg = 'This class is used to calculate some two phase mixture\'s properties, such as '
        msg += 'viscoty, specific volume, void fraction ... '
        return ('What does the class twoPhaseFlowTools_class do? %s'
                '\n--------------------------------------------------------)\n '
                % (msg))

    def volumeEspecificoGas(self, p, T, MMixture):
        '''
        MMixture:  mixture molar weight --> MMixture = np.eisum('i,i', x, MM) [kg/kmol]
        '''
        return ( R * T / (p * MMixture))


    def volumeEspecificaBifasico(self, x, p, T, MMixture, spvolF):
        '''
        MMixture:  mixture molar weight --> MMixture = np.eisum('i,i', x, MM) [kg/kmol] \t
        spvolF: specific volume saturated liquid [m3/kg]
        '''
        spvolG = self.volumeEspecificoGas(p, T, MMixture)
        return ((1.-x) * spvolF + x * spvolG)

       
    def fracaoVazio(self, x, p, T, MMixture, spvolF):
        '''
        x: vapor quality \t
        MMixture:  mixture molar weight --> MMixture = np.eisum('i,i', x, MM) [kg/kmol] \t
        spvolF: specific volume saturated liquid [m3/kg]
        '''
        spvolG = self.volumeEspecificoGas(p, T, MMixture)
        spvolTP = self.volumeEspecificaBifasico(x, p, T, MMixture, spvolF)
        return (spvolG * x / spvolTP)

    def viscosidadeMonofasico(self, xR):
        '''
        Kedzierski & Kaul (1993) correlation - see Guilherme Borges Ribeiro's Thesis (page 51) \t
        Objective: necessary to determine the subcooled liquid's viscosity \t
        xR: vector molar concentration ([xR, xO])
        '''
        G12 = 3.5
        viscO, viscR = self.viscO, self.viscR
        logvisc = np.array([np.log(viscR),np.log(viscO)])
        sum_xlogvisc = np.einsum('i,i', xR, logvisc)
        xRxO_G12 = np.prod(xR) * G12
        return np.exp(sum_xlogvisc + xRxO_G12)
    
    def viscosidadeBifasica(self, x, xR, p, T, MMixture, spvolF):
        '''
        x: vapor quality [-] \t
        xR: vector molar concentration ([xR, xO]) \t
        alfa: void fraction [-] \t
        MMixture:  mixture molar weight --> MMixture = np.eisum('i,i', x, MM) [kg/kmol] \t
        spvolF: specific volume saturated liquid [m3/kg]
         '''
        viscG = self.viscG
        viscF = self.viscosidadeMonofasico(xR)
        alfa = self.fracaoVazio(x, p, T, MMixture, spvolF)
        return (alfa * viscG + viscF * (1. - alfa) * (1. + 2.5 * alfa))  #Eqc 8.33, pag 213 Ghiaasiaan


    def reynoldsBifasico(self, x, xR, p, T, MMixture, spvolF):
        '''
        x: vapor quality [-] \t
        xR: vector molar concentration ([xR, xO])
         '''
        Gt, D = self.Gt, self.D
        viscTP = self.viscosidadeBifasica(x, xR, p, T, MMixture, spvolF)
        return (Gt * D / viscTP)

    