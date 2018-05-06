import numpy as np
from Thermo_pkg.EOS_pkg.EOS_PengRobinson import PengRobinsonEos
from Thermo_pkg.Properties_pkg.Properties import Properties
from Thermo_pkg.Bubble_pkg.BubbleP import Bubble_class
from Thermo_pkg.ThermoTools_pkg import Tools_Convert
from Thermo_pkg.Flash_pkg import FlashAlgorithm_main
from Data_pkg.PhysicalChemistryData_pkg.InputData___ReadThisFile import props
from Thermo_pkg.Departure_pkg.References_Values import input_reference_values_function


R = 8314.  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)


'''
=========================================================================================================
         READING THE FILE DATA 

- specifying (giving names) to the variables that are inside of props list (see InputData_ReadThisFile.py). 
  Check what data you are reading in the InputData_ReadThisFile.py 
  Some data places are left blank; these empty places belong to the variable we want to specify   
                                 
- Reference values of temperature, enthalpy and entropy are collected from Reference_Values.py
=========================================================================================================
'''
(pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props
(TR, hR_mass, sR_mass) = input_reference_values_function()


'''
=================================================================================================================
NECESSARY OBJECTS
=================================================================================================================
'''
eos_obj = PengRobinsonEos(pC, Tc, AcF, omega_a, omega_b, kij)
bubble_obj = Bubble_class(pC, Tc, AcF, kij)
prop_obj = Properties(pC, Tc, AcF, omega_a, omega_b, kij)




'''
=========================================================================================================
         REFERENCES - These references have been used in the departure functions ()
=========================================================================================================

Source: Neto, M. A. M., Barbosa, J. R. Jr, "A departure-function approach to calculate thermodynamic properties
               of refrigerant-oil mixtures", International Journal Of Refrigeration, 36, 2013, (972-979)

The references applied here follows the article above (Moisés). In this article, the SATURATED LIQUID has been adopted:  

The references datas were placed together in the file References_Values.py
TR = 273.15 K (0.0 Celsius)
pR = Pb @ (TR, zi) (You MUST execute BubbleP.py to @ [TR, zi] to set pR) 
hR_mass = 200 kJ/kg [mass base]
sR_mass = 1000 J/(kg K) [mass base]


LEGEND:
AcF: acentric factor [-]
omega_a, omega_b: parameters from Peng Robinson (EdE)
kij: binary interaction factor
Cp: specific heat [J/kmol K]
TR: reference temperature [K]
pR: reference pressure [Pa] -- (You need execute the BubbleP.py, copy pB result and paste here)
T: temperature [K] (Interesting temperature)
Tc: critical temperature [K] 
p: pressure [Pa] (Interesting pressure)
pC: critical pressure [Pa]
z : global_molar_fraction (in bubble saturated condition, xi = zi)
pB: bubble pressure [Pa]
hR_mass: reference enthalpy (in the article was given in mass base, which we need transform to molar base) [J/kg]
hR: reference enthalpy (in molar base) [J/kmol]
sR_mass: reference entropy (in the article was given in mass base, which we need transform to molar base) [J/kg K]
sR: reference entropy (in molar base) [J/kmol K]
MM: molar mass of each component [kg/kmol]
M: mixture molar mass [kg/kgmol] ==> depend on phase composition 
'''



class HSFv:
      '''This class is applied to calculate the enthalpy and entropy. For more detail, print(object's name)
      
      The main method (__call__) returns the following:

      Return: F_V, h, s (vapor quality, enthalpy mass base, entropy mass base)
      '''
      def __init__(self, pC, TR, Tc, AcF, Cp, MM, hR, sR):
            self.pC, self.TR, self.Tc = pC, TR, Tc
            self.AcF, self.Cp, self.MM, self.hR, self.sR = AcF, Cp, MM, hR, sR

      def __str__(self):
            msg = '\n(---------------------------------------------------------)\n'
            msg = 'ENTHALPY/ENTROPY CALCULATIONS: This class HSFv() is useful to calculate the mixture\'s enthalpy and entropy \n'
            msg += 'The evaluation is performed either subcooled or equilibrium liquid-vapor (ELV) \n'
            msg += 'The mixture bubble pressure is always evaluated by the object bubble_obj() at interested \n' 
            msg += 'temperature, T. So, the class evaluates if the system is into subcooled or ELV region. The results are \n'
            msg += 'printed in molar and mass bases (be careful when choose the base).\n' 
            msg += 'The input data and sequence of commands are placed together inside the main() function \n'
            msg += '(---------------------------------------------------------)\n'
            return msg

      def __call__(self, p, T, z):
            pC, TR, Tc = self.pC, self.TR, self.Tc
            AcF, Cp, MM, hR, sR = self.AcF, self.Cp, self.MM, self.hR, self.sR 
            pB, _y_sat, _Sy, _counter = bubble_obj(T, z)
            pR = pB                       #<----setting the reference pressure equal bubble pressure
            if __name__== '__main__':
                  print(self)
                  print('You\'re running pressure = %.3e [Pa] and temperature = %.2f C' % (p, (T - 273.15)))
                  print('For this temperature T = (%.2f C) the bubble pressure is: pB = %.3e [Pa]' % ((T - 273.15), pB))
                  print('\n---------------------------------------------------------)')
            if (p >= pB):
                  F_V = 0.0
                  H_subcooled_molar = prop_obj.calculate_enthalpy(TR, T, pR, p, z, z, hR, Cp, 'liquid')
                  M_L = prop_obj.calculate_weight_molar_mixture(MM, z, 'liquid')
                  h_subcooled_mass = H_subcooled_molar * np.reciprocal(M_L)
                  S_subcooled_molar = prop_obj.calculate_entropy(TR, T, pR, p, z, z, sR, Cp, 'liquid')
                  s_subcooled_mass = S_subcooled_molar * np.reciprocal(M_L)
                  H, h, S, s = H_subcooled_molar, h_subcooled_mass, S_subcooled_molar, s_subcooled_mass
                  if __name__== '__main__':
                        print('--> This @(p,T) falls into single phase region')
                        print('\nSubcooled liquid enthalpy h = %.3e [J/ kmol]' % H)
                        print('\nSubcooled liquid entropy s = %.3e [J/(kmol K)]' % S)
                        print('\nSubcooled liquid entalpy h = %.3e [J/ kg]' % h)
                        print('\nSubcooled liquid entropy s = %.3e [J/(kg K)]' % s)
            else:
                  F_V, _is_stable, K_values_newton, _initial_K_values = \
                  FlashAlgorithm_main.getting_the_results_from_FlashAlgorithm_main(p, T, pC, Tc, AcF, z)
                  x = z / (F_V * (K_values_newton - 1.) + 1.)
                  y = K_values_newton * x
                  x_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, x)
                  y_mass = Tools_Convert.convert_molarfrac_TO_massfrac(MM, y)
                  #Enthalpy
                  H_sat_vapor_molar = prop_obj.calculate_enthalpy(TR, T, pR, p, y, z, hR, Cp, 'saturated_vapor')
                  M_V = prop_obj.calculate_weight_molar_mixture(MM, y, 'saturated_vapor')
                  h_sat_vapor_mass = H_sat_vapor_molar * np.reciprocal(M_V)
                  H_sat_liquid_molar = prop_obj.calculate_enthalpy(TR, T, pR, p, x, z, hR, Cp,'saturated_liquid')
                  M_L = prop_obj.calculate_weight_molar_mixture(MM, x, 'saturated_liquid')
                  h_sat_liquid_mass = H_sat_liquid_molar * np.reciprocal(M_L)
                  H_enthalpy_mixture_molar = (1. - F_V) * H_sat_liquid_molar + F_V * H_sat_vapor_molar
                  F_V_mass = np.reciprocal((M_L / M_V) * (1 / F_V - 1.) + 1.)
                  h_enthalpy_mixture_mass = (1. - F_V_mass) * h_sat_liquid_mass + F_V_mass * h_sat_vapor_mass
                  #Entropy
                  S_sat_vapor_molar = prop_obj.calculate_entropy(TR, T, pR, p, y, z, sR, Cp, 'vapor')
                  s_sat_vapor_mass = S_sat_vapor_molar * np.reciprocal(M_V)
                  S_sat_liquid_molar = prop_obj.calculate_entropy(TR, T, pR, p, x, z, sR, Cp, 'liquid')
                  s_sat_liquid_mass = S_sat_liquid_molar * np.reciprocal(M_L)
                  S_entropy_mixture_molar = (1. - F_V) * S_sat_liquid_molar + F_V * S_sat_vapor_molar
                  s_entropy_mixture_mass = (1. - F_V_mass) * s_sat_liquid_mass + F_V_mass * s_sat_vapor_mass
                  H, h, S, s = H_enthalpy_mixture_molar, h_enthalpy_mixture_mass, S_entropy_mixture_molar, s_entropy_mixture_mass
                  if __name__== '__main__':
                        print('This @(p,T) falls into two phase region')
                        print('The mixture\'s state is ELV with a vapor quality = %.3f [molar base]' % F_V)
                        print('The mixture\'s state is ELV with a vapor quality = %.3f [mass base]' % F_V_mass)
                        print('x [molar base] = ', x)
                        print('y [molar base] = ', y)
                        print('x_mass [mass base] = ', x_mass)
                        print('y_mass [mass base] = ', y_mass)
                        print('\n======\nThe mixture liquid/vapor with h = %.3e [J/ kmol]' % H)
                        print('\n======\nThe mixture liquid/vapor with s = %.3e [J/(kmol K)]' % S)
                        print('The mixture liquid/vapor with h_mass = %.3e [J/ kg] {mass base}' % h)
                        print('The mixture liquid/vapor with s_mass = %.3e [J/(kg K)] {mass base}' % s)
            return F_V, h, s





'''
=========================================================================================================
All the steps are placed together inside the function main()
executing ... 
1st: input data (presure, temperature, molar concentration binary mixture) 
2nd: use the Tools_Convert.frac_input(MM, zin, base) to determine z based on zin (it is just a convertion of base)
3rd: leave the standard values of reference temperature, enthalpy and entropy (but you can change if desire)
4th: change the reference values (3rd step) to molar base
5th: instance of class HSFv() to creat the object hsfv_obj
6th: finally executing the object for the specific variables (p, T, z)

P.S.: it is necessary to set True or False in the main() function: 
      - True: will print results 
      - False: won't prints results
=========================================================================================================
'''



def main():
      #[1] - INPUT DATA
      p = 15e5
      T = 300.
      LC, base = 99./100, 'mass' # <=============================== change here if necessary
      #[2] - CHANGING BASE 
      zin = np.array([LC, (1. - LC)])
      z, _z_mass = Tools_Convert.frac_input(MM, zin, base)
      #[3] - STANDARD VALUES
      hR = hR_mass * prop_obj.calculate_weight_molar_mixture(MM, z, 'saturated_liquid')
      sR = sR_mass * prop_obj.calculate_weight_molar_mixture(MM, z, 'saturated_liquid')
      #[4] - CREATING THE OBJECT
      hsfv_obj = HSFv(pC, TR, Tc, AcF, Cp, MM, hR, sR)
      #[5] - GETTING THE RESULTS
      F_V, h, s = hsfv_obj(p, T, z)


main()