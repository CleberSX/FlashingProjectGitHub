import numpy as np
from EOS_PengRobinson import PengRobinsonEos
from scipy import integrate


R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)



class Properties(PengRobinsonEos):
    '''
    This class joins several methods responsible to calculate important properties like as enthalpy,
    entropy, phase density. It has also auxiliary methods to convert mass fraction to molar fraction (and vice-versa),
    convert specific heat from mass base to molar base, etc.
    '''
    def __init__(self, Pc, Tc, ω, Ω_a, Ω_b, κ_ij):
        super().__init__(Pc, Tc, ω, Ω_a, Ω_b, κ_ij)



    def calculate_weight_molar_mixture (self, MM, x, fluid_type):
        '''
        This method calculate the molar weight of the mixture from molar weight of component "i"

        :x: molar fraction [-]
        :MM: molar weight each component [kg/kmol]
        '''
        return np.einsum('i,i', MM, x)


    def calculate_density_phase(self, p, T, MM, x, fluid_type):
        '''
        This method calculate the phase density (can be vapor ou liquid density)

        :p: pressure [Pa]
        :T: temperature [K]
        :x: molar fraction [-]
        :MM: molar weight each component [kg/kmol]
        :M: molar weight of the mixture/phase [kg/kmol]
        :rho: specific mass of mixture/phase [kg/m3]
        :param fluid_type: put 'vapor' or 'liquid'; it is just to be more clear
        :Z: compressibility factor
        '''
        f, Z = self.calculate_fugacities_with_minimum_gibbs_energy(p, T, x, fluid_type)
        M = self.calculate_weight_molar_mixture(MM, x, fluid_type)
        rho = p*M/(Z*R*T)
        return rho


    def update_parameters(self, p, T, x):
        '''
        This method calculates the variable TdadT and some others (a, A, b, B). These terms appear in...
        ... departure enthalpy and entropy functions involving mixtures

        :Source: Applied Hydrocarbon Thermodynamics - Volume 1
                     Practical Thermodynamics tools for solving process engineering problems
                     Wayne C. Edmister and Byung Ik Lee

                     To see all the variables used, look at pages 60-61 from Edmister's book.


                                     *** [ON PAGES 60-61] ****
           This function evaluate the parameters (a, A, b, B) involved on departure enthalpy (5.85A)...
               ... and entropy (5.94A)

           On Edmister's book the parameter "A" that appears in (5.85A and 5.94A) is given by equation (5.13)...
                        ... and the parameter "a" is given by equation (5.68).
           On Edmister's book the parameter "B" that appears in (5.85A and 5.94A) is given by equation (5.14)...
                        ... and the parameter "b" is given by equation (5.65).
        '''
        self.update_eos_coefficients(p, T, x)
        α, κ_ij, Pc, Tc = self.α_function(T), self.κ_ij, self.Pc, self.Tc
        ω, m, Ω_a, Ω_b = self.ω, self.m, self.Ω_a, self.Ω_b
        acj = Ω_a * (R * Tc)**2 / Pc # [5.10]

        #[TdadT] - Building "TdadT" parameter given by equation [5.82]
        ai = np.einsum('i,i->i', acj, α) #[5.11]
        Trj = T / Tc
        acjTrj = np.einsum('j,j->j', acj, Trj) #[appears in 5.82]
        aiacjTrj = np.einsum('i,j', ai, acjTrj) #[appears in 5.82]
        xjmj = np.einsum('j,j->j', x, m) #[appears in 5.82]
        xixjmj = np.einsum('i,j', x, xjmj) #[appears in 5.82]
        xixjmj_SQRT_aiacjTrj = np.einsum('ij,ij->ij', xixjmj, np.sqrt(aiacjTrj)) #[appears in 5.82]
        self.TdadT = - np.einsum('ij,ij->', xixjmj_SQRT_aiacjTrj, (1-κ_ij)) #[5.82]

        #[a] - Building parameter "a" given by equation [5.68] - pg 60
        SQRTaiaj = np.sqrt( np.einsum('i,j', ai, ai) ) #[appears in 5.68]
        SQRTaiaj_kij = SQRTaiaj * (1.0 - κ_ij)  # (m,n) #[appears in 5.68]
        xjSQRTaiaj_kij = np.einsum('j,ij', x, SQRTaiaj_kij)  # (m,) vector #[appears in 5.68]
        self.a = np.einsum('i,i', x, xjSQRTaiaj_kij)  # scalar #[5.68]

        # [b] - Building parameter "b" given by equation [5.65] - pg 60
        bi = Ω_b * (R * Tc) / Pc  # [5.8]
        self.b = np.einsum('i,i', x, bi)  # scalar

        #[A & B] - Building parameters A and B (equations 5.13 and 5.14, respectively)
        self.A = self.a * p / (R * T) ** 2
        self.B = self.b * p / (R * T)


    def calculate_departure_enthalpy(self, p, T, x, fluid_type):
        '''
            :Source: Applied Hydrocarbon Thermodynamics - Volume 1
                     Practical Thermodynamics tools for solving process engineering problems
                     Wayne C. Edmister and Byung Ik Lee

            This function calculate_departure_enthalpy was based on equation (5.85A) - Edmister

        :T: temperature [K]
        :x: molar fraction [-]
        :p: pressure [Pa]
        '''
        self.update_parameters(p, T, x)
        f, Z = self.calculate_fugacities_with_minimum_gibbs_energy(p, T, x, fluid_type)
        # print('Considering T = %.2f and P = %.2e ==> valor of Z = %.5f to %s phase' % (T, p, Z, fluid_type))
        h_i = (Z - 1.0)
        h_ii = self.A / (2 * SQRT_2 * self.B)
        h_iii = (1.0 - self.TdadT / self.a)
        h_iv = (Z + (SQRT_2 + 1.0) * self.B) / (Z - (SQRT_2 - 1.0) * self.B)
        return R * T * (h_i - h_ii * h_iii * np.log( h_iv ))


    def calculate_departure_entropy(self, p, T, x, fluid_type):
        '''
        :Source: Applied Hydrocarbon Thermodynamics - Volume 1
                     Practical Thermodynamics tools for solving process engineering problems
                     Wayne C. Edmister and Byung Ik Lee

           This function calculate_departure_entropy was based on equation (5.94A) - Edmister

        :T: temperature [K]
        :x: molar fraction [-]
        :p: pressure [Pa]
        :p0: ambient pressure which is set equal to 101325 [Pa]
        '''
        p0 = 101325.
        self.update_parameters(p, T, x)
        f, Z = self.calculate_fugacities_with_minimum_gibbs_energy(p, T, x, fluid_type)
        s_i = np.log(Z - self.B)
        s_ii = self.A / (2 * SQRT_2 * self.B) * (1. - self.TdadT / self.a )
        s_iii = (Z + (SQRT_2 + 1.0) * self.B ) / (Z - (SQRT_2 - 1.0) * self.B )
        return R * ( s_i - s_ii * np.log( s_iii ) - np.log(p / p0) )


    def calculate_enthalpy_ig_diff(self, T, x, Cp):
        '''
        This method calculates the differential Cp_mixture*dT, where Cp_mixture = sum(xi * Cpi)

        :T: temperature [K]; even though temperature doesn't appear in the equation, it is necessary because
                           it is integration variable in the following function ==> calculate_enthalpy_int()
        :x: molar fraction [-]
        :Cp: specific_heat is in molar base [J / kmol K]

        (===== Obs.: be careful if the specific heats are in molar or mass base; if they are in mass base, first
        you must use the method "convert_specific_heat_massbase_TO_molarbase()" to convert them to molar base ====)
        '''
        return np.einsum('i,i', x, Cp)



    def calculate_enthalpy_ig_int(self, TL, TH, x, Cp):
        '''
        This method calculates the integration of the differential equation "calculate_enthalpy_ig_diff()"

        :TL: the temperature low_limit of integration [K]
        :TH: the temperature high_limit of integration [K]
        :x: molar fraction (can be feed composition OR vapor phase composition OR liquid phase composition) [-]
        :Cp: specific_heat in molar base [J / kmol K]

        (===== Obs.: be careful if the specific heats are in molar or mass base; if they are in mass base, first
        you must use the method "convert_specific_heat_massbase_TO_molarbase()" to convert them to molar base ====)
        '''
        return integrate.quad(self.calculate_enthalpy_ig_diff, TL, TH, args=(x, Cp))


    def calculate_entropy_ig_diff(self, T, x, Cp):
        '''
        This method calculate the differential (Cp_mixture / T) * dT, where Cp_mixture = sum(x * Cpi)

        :x: molar fraction (can be feed composition OR vapor phase composition OR liquid phase composition) [-]
        :Cp: specific_heat in molar base [J / kmol K]
        :T: temperature [K]

        (===== Obs.: be careful if the specific heats are in molar or mass base; if they are in mass base, first
        you must use the method "convert_specific_heat_massbase_TO_molarbase()" to convert them to molar base ====)
        '''
        return ( np.einsum('i,i', x, Cp) / T )

    def calculate_entropy_ig_int(self, TL, TH, x, Cp):
        '''
        This method calculate the integration of the differential equation "calculate_entropy_ig_diff()"

        :TL: the temperature low_limit of integration [K]
        :TH: the temperature high_limit of integration [K]
        :x: molar fraction (can be feed composition OR vapor phase composition OR liquid phase composition) [-]
        :Cp: specific_heat in molar base [J / kmol K]

        (===== Obs.: be careful if the specific heats are in molar or mass base; if they are in mass base, first
        you must use the method "convert_specific_heat_massbase_TO_molarbase()" to convert them to molar base ====)
        '''
        return integrate.quad(self.calculate_entropy_ig_diff, TL, TH, args=(x, Cp))



    def calculate_enthalpy(self, TR, T, pR, p, x, z, hR, Cp, fluid_type):
        '''
        This method calculates the enthalpy of the mixture using Departure Enthalpy. Even though the equations
        used to evaluate departure functions were built up based on Edmister's book, the final equation
         to calculate the mixture enthalpy was taken from article below: see Eq (10)

        Source: Neto, M. A. M., Barbosa, J. R. Jr, "A departure-function approach to calculate thermodynamic properties
               of refrigerant-oil mixtures", International Journal Of Refrigeration, 36, 2013, (972-979)

        :href_mass (reference_specific_enthalpy_mass [J/kg]): this is the specific reference enthalpy;
                                                              for refrigeration, according with the International
                                                              Institute of Refrigeration (IIR), the value
                                                              is hr = 200.000 J/kg

        :href (reference_specific_enthalpy [J/kmol]): is the href_mass, but converted to molar base


        :TR (reference_temperature [K]) and pR (reference_pressure [Pa]): are temperature and pressure of reference
        :MM (molar_mass [kg/kmol]) = molar weight of component "i"

        :x (molar_fraction[-]): is the phase concentration em molar base (x => x when liquid phase;...
                              ... x => y when vapor phase)
        :z: global molar fraction [-] -- represents the feed composition
        :Cp (specific_heat [J/ kmol K]): specific_heat of component "i" in molar base
        :Dep_h: departure enthalpy at isotherm you're interested, i.e., Dep_h = f(p,T,x)
        :Dep_hR: departure enthalpy at reference isotherm, i.e, Dep_hR = f(pR,TR,z)

        (=====
        Obs.: be careful if the specific heats are in molar or mass base; if they are in mass base, first
        you must use the method "convert_specific_heat_massbase_TO_molarbase()" to convert them to molar base
        ======)
        '''
        Dep_h = self.calculate_departure_enthalpy(p, T, x, fluid_type)
        Dep_hR = self.calculate_departure_enthalpy(pR, TR, z, 'saturated liquid')
        h_cpterm, errorH = self.calculate_enthalpy_ig_int(TR, T, z, Cp)
        # print('Residual enthalpy @ (pR, TR) = %.2e' % Dep_hR )
        # print('Residual enthalpy @ (p,T) = %.2e' % Dep_h)
        # print('Termo GI in enthalpy = %.2e' % h_cpterm)
        return (hR + Dep_h + h_cpterm - Dep_hR)


    def calculate_entropy(self, TR, T, pR, p, x, z, sR, Cp, fluid_type):
        '''
        This method calculates the entropy of the mixture using Departure Entropy. Even though the equations
        used to evaluate departure functions were built up based on Edmister's book, the final equation
         to calculate the mixture entropy was taken from article below: see Eq (11)

        (=====
        Obs.: Eq (11) has a term R*sum(x*ln x) -->  I think this term doesn't exist. And you can check using,...
        =====)                                      for example, the T = 273.15 K and pBubble = p(T) with the
                                                    REFPROP. The result obtained with software only match with
                                                    this Python code if we don't use this term R*sum(x*ln x)



        Source: Neto, M. A. M., Barbosa, J. R. Jr, "A departure-function approach to calculate thermodynamic properties
               of refrigerant-oil mixtures", International Journal Of Refrigeration, 36, 2013, (972-979)

        :sref_mass (reference_specific_entropy_mass [J/(kg K)]): this is the specific reference entropy;
                                                              for refrigeration, according with the International
                                                              Institute of Refrigeration (IIR), the value
                                                              is sr = 1000 J/kg;

        :sref (reference_specific_entropy [J/(kmol K)]): is the sref_mass, but converted to molar base


        :TR (reference_temperature [K]) and pR (reference_pressure [Pa]): are temperature and pressure of reference
        :MM (molar_mass [kg/kmol]) = molar weight of component "i"

        :x (molar_fraction[-]): is the phase concentration em molar base (x => x when liquid phase;...
                              ... x => y when vapor phase)
        :z: global molar fraction [-] -- represents the feed composition
        :Cp (specific_heat [J/ (kmol K)]): specific_heat of component "i" in molar base
        :Dep_s: departure entropy at isotherm you're interested, i.e., Dep_s = f(p,T,x)
        :Dep_sR: departure entropy at reference isotherm, i.e, Dep_sR = f(pR,TR,z)

        (=====
        Obs.: be careful if the specific heats are in molar or mass base; if they are in mass base, first
        you must use the method "convert_specific_heat_massbase_TO_molarbase()" to convert them to molar base
        ======)
        '''
        Dep_s = self.calculate_departure_entropy(p, T, x, fluid_type)
        Dep_sR = self.calculate_departure_entropy(pR, TR, z, 'saturated liquid')
        s_cpterm, errorS = self.calculate_entropy_ig_int(TR, T, z, Cp)
        # print('Residual entropy @ (pR, TR) = %.2e' % Dep_sR )
        # print('Residual entropy @ (p,T) = %.2e' % Dep_s)
        # print('Termo GI in entropy = %.2e' % s_cpterm)
        return ( sR + Dep_s + s_cpterm - R * np.log(p / pR) - Dep_sR )


    def __str__(self):
        explanation = ('This class is used to calculate some mixture\'s properties, such as enthalpy '
                       'and entropy. The method used was based on Departure Functions according to '
                       'Elliot and Lira, 2nd, (2012)')
        enthalpy = ('hR + Dep_h + h_cpterm - Dep_hR')
        entropy = ('sR + Dep_s + s_cpterm - Dep_sR - R * np.log(p / pR)')
        return ('What does this class do? %s'
                '\nh = %s '
                '\ns = %s '
                '\n--------------------------------------------------------)\n '
                % (explanation, enthalpy, entropy))
