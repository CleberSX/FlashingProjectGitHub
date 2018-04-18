import numpy as np 
from CoolProp.CoolProp import PropsSI

R = 8314.34                         # J / (kmol K)


class FlowTools_class():
    '''THIS CLASS IS NECESSARY TO CALCULATE FLUID PROPERTIES: SINGLE PHASE & TWO PHASE \t
    D: diameter [m] \t
    Gt: superficial total mass flux [(kg/s)/m2]
    '''
    def __init__(self, D, Gt):
        self.D, self.Gt = D, Gt
    
    
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
        x: vapor quality \t
        MMixture:  mixture molar weight 
        spvolF: specific volume saturated liquid [m3/kg]
        '''
        spvolG = self.volumeEspecificoGas(p, T, MMixture)
        return ((1.-x) * spvolF + x * spvolG)

       
    def fracaoVazio(self, x, p, T, MMixture, spvolF):
        '''
        x: vapor quality \t
        MMixture:  mixture molar weight 
        spvolF: specific volume saturated liquid [m3/kg]
        '''
        spvolG = self.volumeEspecificoGas(p, T, MMixture)
        spvolTP = self.volumeEspecificaBifasico(x, p, T, MMixture, spvolF)
        return (spvolG * x / spvolTP)

    def viscO_function(self, T):

        '''
        This function calculate the POE ISO VG 10 viscosity \n
    
        T: temperature [K] \n
        Oil Viscosity: in [Pa.s] \n


        This correlation has been gotten from: \n 
        Tese de doutorado do Dalton Bertoldi (2014), page 82 \n

        "Investigação Experimental de Escoamentos Bifásicos com mudança \n
        de fase de uma mistura binária em tubo de Venturi" \n
         '''
        Tcelsius = T - 273.15
        return 0.04342 * np.exp(- 0.03529 * Tcelsius)

    def viscR_function(self, T, p):
        '''
        This function is call the CoolProp
        T: temperature [K] 
        p: pressure [Pa]
        Refrigerant's dynamic viscosity [Pa.s] \t
        '''
        return PropsSI("V", "T", T, "P", p,"R134a")


    def viscosidadeMonofasico(self, T, p, xR, G12 = 3.5):
        '''
        GRUNBERG & NISSAN (1949) correlation - see Dalton Bertoldi's Thesis (page 81) \n
        Objective: necessary to determine the subcooled liquid's viscosity \n
        T: temperature [K] \n
        p: pressure [Pa] \n
        xR: vector molar concentration ([xR, xO]) \n
        G12: model's parameter (G12 = 3.5 has been taken from Dalton's Thesis (page 82)) \n
        Refrigerant viscosity is get from CoolProp \n
        viscO: Oil's dynamic viscosity [Pa.s] \n
        viscR: Refrigerant's dynamic viscosity [Pa.s]
        '''
        viscO = self.viscO_function(T)
        viscR = self.viscR_function(T, p)
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
        viscG = 12e-6 #valor qqer...temporário...preciso entrar com uma equação aqui
        viscF = self.viscosidadeMonofasico(T, p, xR)
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

    