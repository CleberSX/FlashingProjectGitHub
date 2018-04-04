import numpy as np 

R = 8314.34                         # J / (kmol K)


class twoPhaseFlowTools_class():
    def __init__(self, MM, viscG, viscF, D, Gt):
        self.MM, self.viscG, self.viscF, self.D, self.Gt = MM, viscG, viscF, D, Gt
    
    
    def __str__(self):
        msg = 'This class is used to calculate some two phase mixture\'s properties, such as '
        msg += 'viscoty, specific volume, void fraction ... '
        return ('What does the class twoPhaseFlowTools_class do? %s'
                '\n--------------------------------------------------------)\n '
                % (msg))

    def volumeEspecificoGas(self, p, T, MMixture):
        return ( R * T / (p * MMixture))


    def volumeEspecificaBifasico(self, x, p, T, MMixture, spvolF):
        spvolG = self.volumeEspecificoGas(p, T, MMixture)
        return ((1.-x) * spvolF + x * spvolG)

       
    def fracaoVazio(self, x, p, T, MMixture, spvolF):
        spvolG = self.volumeEspecificoGas(p, T, MMixture)
        spvolTP = self.volumeEspecificaBifasico(x, p, T, MMixture, spvolF)
        return (spvolG * x / spvolTP)


    
    def viscosidadeBifasica(self, x, p, T, MMixture, spvolF):
        viscG, viscF = self.viscG, self.viscF
        alfa = self.fracaoVazio(x, p, T, MMixture, spvolF)
        return (alfa * viscG + viscF * (1. - alfa) * (1. + 2.5 * alfa))  #Eqc 8.33, pag 213 Ghiaasiaan


    def reynoldsBifasico(self, x, p, T, MMixture, spvolF):
        Gt, D = self.Gt, self.D
        viscTP = self.viscosidadeBifasica(x, p, T, MMixture, spvolF)
        return (Gt * D / viscTP)