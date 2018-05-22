import numpy as np 
import sys
from scipy import optimize
from Thermo_pkg.ThermoTools_pkg import Tools_Convert
from Thermo_pkg.Properties_pkg.Properties import Properties
from CoolProp.CoolProp import PropsSI

R = 8314.34                         # J / (kmol K)


class FlowTools_class(Properties):
    '''THIS CLASS IS NECESSARY TO CALCULATE FLUID PROPERTIES: SINGLE PHASE & TWO PHASE \n
    D: diameter [m] \n
    mdotL: subcooled mass rate [kg/s] \n
    Gt: superficial total mass flux [(kg/s)/m2] \n
    '''
    

    def __init__(self, Pc, Tc, ω, Ω_a, Ω_b, κ_ij, mdotL):
        super().__init__(Pc, Tc, ω, Ω_a, Ω_b, κ_ij)
        self.mdotL = mdotL
    
    
    
    def __str__(self):
        msg = 'This class is used to calculate some two phase mixture\'s properties, such as '
        msg += 'viscoty, specific volume, void fraction ... '
        msg += 'methods with .SEC at the their name end are secundary methods (work in backstage)' 
        return ('What does the class twoPhaseFlowTools_class do? %s'
                '\n--------------------------------------------------------)\n '
                % (msg))

    def specificVolumGas(self, p, T, MM, x):
        '''
        MMixture:  mixture molar weight --> MMixture = np.eisum('i,i', x, MM) [kg/kmol]
        '''

        rhoG = self.calculate_density_phase(p,T,MM,x,'vapor')
        if rhoG == 0.0: volG = 0.0
        else: volG = np.power(rhoG, -1)

        return volG


    def specificVolumeTwoPhase(self, q, volG, volF):
        '''
        q: vapor quality \n

        volG: specific volume saturated vapor [m3/kg] \n
        volF: specific volume saturated liquid [m3/kg] \n
        volTP: specific volume two phase [m3/kg]

        Return: volTP
        '''
        return ((1.-q) * volF + q * volG)

       
    def voidFraction(self, q, volG ,spvolTP):
        '''
        q: vapor quality \n
        MMixture:  mixture molar weight \n
        spvolF: specific volume saturated liquid [m3/kg] \n

        Return: void_fraction
        '''
        return (volG * q / spvolTP)


    def densityLiquid_jpDias_SEC(self, T, p, x_mass):
        ''' 
        The correlation was copied from JP Dias's Thesis (pg 294, EQ A.2) \n

        ESCOAMENTO DE ÓLEO E REFRIGERANTE PELA FOLGA PISTÃO-CILINDRO DE 
        COMPRESSORES HERMÉTICOS ALTERNATIVOS (2012) - UFSC \n

        T: temperature [K] \n
        p: pressure [Pa] \n
        x_mass: vector mass concentration [-] (x_mass = ([xR_mass, xO_mass])) \n

        This correlation is valid only in interval 20C < temp. celsius < 120C \n

        Return: densL
        '''
        wr = x_mass[0]
        Tc = T - 273.15

        densO = 966.43636 - 0.57391608 * Tc - 0.00024475524 * Tc ** 2
        densR = PropsSI("D", "T", T, "P", p,"R134a")
        densL = densO * np.power( (1. + wr * (densO / densR - 1.) ), -1)
        return densL




    def specificVolumeLiquid_Wrap(self, p, T, MM, x, density_model='jpDias'):
        '''
        This function chooses density from EXperimental CORrelation ('jpDias') or from THErmodynamics ('ELV') \n

        T: temperature [K] \n
        p: pressure [Pa] \n
        MM: vector with molar weight of each component "i" [kg/kmol], i.e., (MM = ([MMrefrig, MMpoe])) 
        x: vector molar concentration [-], i.e., (x = ([xR, xO])) \n
        x_mass: vector mass concentration [-] (x_mass = ([xR_mass, xO_mass])) \n
        spvolL: specific volume from correlation 'jpDias' or from 'ELV' [m3/kg] \n

        Return: spvolL'''

        nome_desta_funcao = sys._getframe().f_code.co_name
        x_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, x)
        density_models = ['ELV', 'jpDias']
        if density_model not in density_models:
            msg = 'Invalid density model in --> %s' % nome_desta_funcao
            msg += '\t Choose one of the models: %s' % density_models
            raise Exception(msg)
        if density_model == 'ELV':
            densL = super().calculate_density_phase(p, T, MM, x, fluid_type='liquid')
        elif density_model == 'jpDias':
            densL = self.densityLiquid_jpDias_SEC(T, p, x_mass)
        spvolL = np.power(densL, -1) 
        return spvolL

    def specificLiquidHeat_jpDias(self, T, p, x_mass):
        ''' 
        The correlation was copied from JP Dias's Thesis (pg 295, EQ A.7) \n

        ESCOAMENTO DE ÓLEO E REFRIGERANTE PELA FOLGA PISTÃO-CILINDRO DE 
        COMPRESSORES HERMÉTICOS ALTERNATIVOS (2012) - UFSC \n

        T: temperature [K] \n
        p: pressure [Pa] \n
        x_mass: vector mass concentration [-] (x_mass = ([xR_mass, xO_mass])) \n

        This correlation is valid only in interval: ? \n
        

        Return: cpL [J/kg K] (?...tenho verificar se são essas as unidades!!)
        '''
        Tc = T - 273.15
        wr = x_mass[0]
        
        CpR = PropsSI("Cpmass", "T", T, "P", p,"R134a")
        CpO = 2411.5968 + 2.260872 * Tc
        return (1. - wr) * CpO + wr * CpR

    


    def viscosityLiquid_jpDias_SEC(self, T, p, x_mass):

        ''' 
        The correlation was copied from JP Dias's Thesis (pg 294, EQ A.4) \n

        ESCOAMENTO DE ÓLEO E REFRIGERANTE PELA FOLGA PISTÃO-CILINDRO DE 
        COMPRESSORES HERMÉTICOS ALTERNATIVOS (2012) - UFSC \n

        T: temperature [K] \n
        p: pressure [Pa] \n
        x_mass: vector mass concentration [-] (x_mass = ([xR_mass, xO_mass])) \n

        This correlation is valid only in interval: \n
        0 < temp. celsius < 120C and 0.0 < refrigerant < 50.0% \n

        Return: viscL [Pa.s]
        '''

        Tc = T - 273.15
        wr = x_mass[0] * 100

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
        densL = self.densityLiquid_jpDias_SEC(T, p, x_mass)
        return viscCinem * densL * 1e-6


    def viscosityLiquid_NISSAN_SEC(self, T, p, x, G12 = 3.5):
        '''
        GRUNBERG & NISSAN (1949) correlation - see Dalton Bertoldi's Thesis (page 81) \n
        Objective: necessary to determine the subcooled liquid's viscosity \n
        T: temperature [K] \n
        p: pressure [Pa] \n
        x: vector molar concentration x = ([xR, xO]) \n
        G12: model's parameter (G12 = 3.5 has been taken from Dalton's Thesis (page 82)) \n
        Refrigerant viscosity is get from CoolProp \n
        viscO: POE ISO VG 10 viscosity [Pa.s] \n
        viscR: R134a's dynamic viscosity [Pa.s]

        ==============================================================================
        ViscO correlation has been gotten from: \n 
        Tese de doutorado do Dalton Bertoldi (2014), page 82 \n

        "Investigação Experimental de Escoamentos Bifásicos com mudança \n
        de fase de uma mistura binária em tubo de Venturi" \n
        '''
        Tc = T - 273.15
        viscO = 0.04342 * np.exp(- 0.03529 * Tc)
        viscR = PropsSI("V", "T", T, "P", p,"R134a")
        logvisc = np.array([np.log(viscR),np.log(viscO)])
        sum_xlogvisc = np.einsum('i,i', x, logvisc)
        xRxO_G12 = np.prod(x) * G12
        return np.exp(sum_xlogvisc + xRxO_G12)


    def viscosityLiquid_Wrap(self, p, T, x, x_mass, visc_model='jpDias'):
        '''
        This function/method choose the fluid single phase viscosity \n
        
        For while, there are just two options for liquid viscosity: 'jpDias' correlation or 'NISSAN' model \n

        For more informations about 'jpDias' and 'NISSAN' you must read jpDias_liquidViscositySEC() and 

        T: temperature [K] \n
        p: pressure [Pa] \n
        x: vector molar concentration x = ([xR, xO]) \n
        x_mass: vector mass concentration [-] (x_mass = ([xR_mass, xO_mass])) \n
        visc_model: 'jpDias' or 'NISSAN' \n
        viscL: liquid viscosity [Pa.s] \n

        Return: viscL
        '''
        nome_desta_funcao = sys._getframe().f_code.co_name

        visc_models = ['jpDias', 'NISSAN']
        if visc_model not in visc_models:
            msg = 'Invalid viscosity model inside function: %s' % nome_desta_funcao
            msg += '\t Choose one of the models: %s' % visc_models
            raise Exception(msg)
        if visc_model == 'NISSAN':
            viscL = self.viscosityLiquid_NISSAN_SEC(T, p, x)
        elif visc_model == 'jpDias':
            viscL = self.viscosityLiquid_jpDias_SEC(T, p, x_mass)
        return viscL




    def reynolds_function(self, Gt, Dc, viscL):
        '''
        This function/method calculates Reynolds number \n
        
            The liquid viscosity depend of the model you've been chosen \n
            For while, there are just two options for liquid viscosity \n

        Gt: mass flux [kg/s.m2] \n
        Dc: diameter [m] \n
        Re: Reynolds number [-] \n

        Return: Re
        '''

        return Gt * Dc / viscL

    


    def frictionChurchillSEC(self, Re, ks, Dc):
        '''
        Churchill equation to estimate Fanning friction factor, f_F, (pg 149, Ron Darby's book) \n
            Can be applied for all flow regimes in single phase \n
            Re: Single phase Reynolds number [-] \n
            ks: rugosity [m] \n
            Dc: diameter [m] \n

        Return: f0 (First estimative for Fanning frictions factor)
        '''
        f1 = 7. / Re
        f2 = 0.27 * ks/Dc
        a = 2.457 * np.log(1. / (f1 ** 0.9 + f2))
        A = a ** 16
        b = 37530. / Re
        B = b ** 16
        f3 = 8. / Re
        return  2 * (f3 ** 12 + 1./(A + B) ** 1.5 ) ** (1. / 12)

    def frictionColebrookSEC(self, Re, ks, Dc):  
        '''This Colebrook function determines the Fanning friction factor \n

            In fluid mechanis Colebrook calculates Darcy factor. Here we convert it to Fanning factor \n
            For Re < 4000 Fanning factor is 16/Re (we added a IF statement for laminar flow case)  \n

        Return: f_F
        '''  
        colebrook = lambda f0: - 2. * np.log10( ks / (3.7 * Dc) + 2.51 / (Re * np.sqrt(f0)) ) - 1. / np.sqrt(f0)
        f0 = 0.25 * (np.log( (ks / Dc) / 3.7 + 5.74 / Re ** 0.9 )) ** (-2) #Fox, pg 234, EQ 8.37b
        f_D = optimize.newton(colebrook, f0)  #Darcy 
        if Re > 2300.: f_F = f_D / 4.
        else: f_F = 16. / Re
        return f_F
    
    
    def frictionFactorFanning_Wrap(self, Re, ks, Dc, friction_model='Colebrook'):
        '''This is the function/method we must call to calculate Fanning friction factor'''

        nome_desta_funcao = sys._getframe().f_code.co_name

        friction_models = ['Colebrook', 'Churchill']
        if friction_model not in friction_models:
            msg = 'Invalid friction model inside the function: %s' % nome_desta_funcao
            msg += '\t Choose one of the models: %s' % friction_models
            raise Exception(msg)
        if friction_model == 'Churchill':
            f_F = self.frictionChurchillSEC(Re, ks, Dc)
        elif friction_model == 'Colebrook':
            f_F = self.frictionColebrookSEC(Re, ks, Dc)
        return f_F




    def viscosityTwoPhase(self, q, volG, volTP, viscF, viscTP_model, viscG=12e-6):
        '''
        q: vapor quality [-] \n
        viscF: liquid phase viscosity [Pa.s]
        viscg: vapor phase viscosity [Pa.s]
        viscTP: two-phase viscosity [Pa.s] \n
        viscTP_model: models of two-phase flow (McAdams, Cicchitti, Dukler)

        Return: viscTP
         '''
        nome_desta_funcao = sys._getframe().f_code.co_name

        viscTP_models = ['McAdams', 'Cicchitti', 'Dukler']
        if viscTP_model not in viscTP_models:
            msg = 'Invalid viscosity model for the two phase flow inside the function: %s' % nome_desta_funcao
            msg += '\t Choose one of the models: %s' % viscTP_models
            raise Exception(msg)

        if viscTP_model == 'McAdams':
            viscTP = (q / viscG + (1. - q) / viscF) ** (-1)
        elif viscTP_model == 'Cicchitti':
            viscTP = q * viscG + (1. - q) * viscF
        elif viscTP_model == 'Dukler':
            beta = q * volG / volTP
            viscTP = beta * viscG + (1. - beta) * viscF

        return viscTP


    def reynoldsBifasico(self, Gt, Dc, viscTP):
        '''
        Gt: mass flux [kg/s.m2] \n
        Dc: diameter [m] \n
        viscTP: two-phase viscosity [Pa.s] \n
        ReTP: two phase Reynolds [-] \n

        Return: ReTP
         '''

        return (Gt * Dc / viscTP)


    def twoPhaseMultiplier(self, q, viscTP, viscL, volG, volL, n=-0.25):
        '''
        :param q: vapor quality [-]
        viscTP: two-phase viscosity model [Pa.s] \n
        viscL: liquid phase viscosity [P.s] \n
        viscG: vapor phase viscosity [P.s] \n
        volG: gas phase specific volume [m3/kg] \n
        volL: liquid phase specific volume [m3/kg] \n
        n: model parameter in two-phase multiplier (n=-0.25 when Blasius is used) \n

        phiLO2: two-phase multiplier [-]

        :Return: phiLO2
        '''

        phiLO2 = np.power((viscTP / viscL), n) * (1. + (volG / volL - 1.) * q)

        return phiLO2





def solution_concentration_set_function(lightComponent: object, MM: object, base: object) -> object:
    '''
    This function is useful to obtain mass fraction and molar fraction VECTOR of a BINARY mixture \n


    lightComponent: the subcooled liquid solution concentration is set up ...
                ... based on R134a concentration (e.g.: 5./100, 0.05/100, etc...) \n
    MM: vector with molar weight of each component "i" [kg/kmol], i.e., (MM = ([MMrefrig, MMpoe])) \n
    base: 'molar' or 'mass' [-]; you need inform one of them \n
    z: vector concentration [-], i.e., (z = ([zR, zO])) \n
    z_mass: vector mass concentration [-] (z_mass = ([zR_mass, zO_mass])) \n

    Return: z, z_mass
    '''
    nome_desta_funcao = sys._getframe().f_code.co_name
    if lightComponent > 1. or lightComponent < 0.0:
        raise Exception('You entered with a concentration out of expected range whe you calls the function ' + str(nome_desta_funcao))
    elif MM.shape[0] != 2:
        raise Exception('You must entered a vector like MM = ([MM1,MM2]) when you calls the function '
                        + str(nome_desta_funcao))
    elif base != 'mass' and 'molar':
        raise Exception('You need type \'mass\' or \'molar\' when calls the function: ' + str(nome_desta_funcao))

    zin = np.array([lightComponent, (1. - lightComponent)])
    z, z_mass = Tools_Convert.frac_input(MM, zin, base)
    return z, z_mass