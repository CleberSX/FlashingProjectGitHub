
import numpy as np
import Tools_Convert, BubbleP


R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)


'''
=========================================================================================================
    O QUE ESTA FUNCAO FAZ? 
    
                       CALCULA A PRESSAO DE BOLHA PARA UMA ISOTERMA & CONCENTRACAO, OU SEJA, pB = pB(T,z).
=========================================================================================================


Pontos que devemos alterar/alimentar:
(1) - verificar qual arquivo (dados do problema) está sendo importado no arquivo ==> InputData__ReadThisFile.py
(2) - atencao na faixa pressure_low e pressure_high (sao os limites de busca)
(3) - o step eh o tamanho do passo (não é a quantidade de passos). Assim, quanto menor este número, maior ...
      ... o número de pontos avaliados
(4) - IMPORTANT: YOU MUST CHECK the temperature (T) and global_molar_fraction (z) (write below the values
                 ... you are interested to evaluate)
'''



'''
=========================================================================================================
        CHANGE HERE (I)

        Building the isotherm and molar concentration to feed the code. Why does it necessary?
        
        Because Pb = Pb @(T, z): We want Pb for a specific global molar fraction and isotherm 
=========================================================================================================

LEGEND:
T: Interested temperature (evaluate the enthalpy considering this isotherm) [K]
LC: Lighter component of your binary mixture
base: It is the base of your fraction, i.e., you must specify 'molar' ou 'mass'
z_mass: global {mass} fraction 
z: global {molar} fraction
'''
T = 520. #(40. + 273.15)   # <=================================== change here
LC, base = 0.10, 'mass' # <=============================== change here
zin = np.array([LC, (1. - LC)])
z, z_mass = Tools_Convert.frac_input(BubbleP.MM, zin, base)



'''
=========================================================================================================
        CHANGE HERE (II)

        BUILDING A RANGE: this 4 lines below it is necessary to create a range of pressure necessary in...
                          ... executable() function.
                          To do this it's necessary to import the function calculate_pressure_guess()...
                          ... from BubbleP.py
                          
                          VERY IMPORTANT!: the step size determines how close we are to the correct root. So,
                                           smaller step produces more precise results  
=========================================================================================================

LEGEND:
pG: Guess pressure [Pa] - It is necessary to initialize the search for correct bubble pressure  
pL and pH: Low pressure and High pressure [Pa] - They are necessary to build a window/range of pressure for the...
           ...numerical method begins the search the correct bubble pressure in Bubble.py
step: This is the step used by the numerical method in Bubble.py (very important)
'''
pG = BubbleP.calculate_pressure_guess(z, T, BubbleP.pC, BubbleP.Tc, BubbleP.AcF)
# pG = 3.9e+06 #2.16014935e+05
pL = 0.5 * pG
pH = 1.5 * pG
step = 1000               # <=================================== change here



'''
=========================================================================================================
  EXECUTING: executing the imported function executable() from BubbleP.py to get the bubble pressure
=========================================================================================================

P.S.: note that liquid composition, x,  is set equal to global molar composition, z, for problems involving bubble P

LEGEND:
pB: Bubble pressure [Pa]   
Sy: Sum of molar fraction vapor phase [-]
yi: Molar fraction of vapor phase [-] 
'''
pB = BubbleP.executable(step, pL, pH, T, z)
Sy, yi = BubbleP.function_inner_loop(pB, T, z, stability=True)



'''
=========================================================================================================
  CHECKING: this if/elif is useful to check if you're using acceptable limits (built above)
=========================================================================================================
'''
if ((pB / pL) < 1.001):
    print('ATTENTION: the pB can be out of LOW limit (pL) you have set')
elif ((pB / pH) > 0.98):
    print('ATTENTION: the pB can be out of HIGH limit (pH) you have set')



'''
=========================================================================================================
  PRINT RESULTS:
=========================================================================================================
'''
print('\n---------------------------------------------------')
print('[1] - Guess pB [Pa]= %.8e' % pG)
print('[2] - Seek the pB [MPa] in --> [LL = %.4e & HL = %.4e] using STEP = %i' % (pL, pH, step) + '\n')
print('[3] - ======> at T = %.2f [C], pB = %.8e [Pa] ' % ((T-273.15), pB) + '\n')
print('[4] - Concentration vapor phase [molar] = ', yi.round(3))
print('[5] - Concentration vapor phase [mass] = ', Tools_Convert.convert_molarfrac_TO_massfrac(BubbleP.MM,yi))
print('[6] - Pay attention if Sy is close to unity (Sy = %.10f)' % Sy)
print('[7] - Global {mass} fraction = ', z_mass.round(3))
print('[8] - Global {molar} fraction = ', z.round(3))





