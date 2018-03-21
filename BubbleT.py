'''
ESTA FUNCAO CALCULA A TEMPERATURA DE BOLHA.
PONTOS QUE DEVEMOS ALIMENTAR/ALTERAR:
(1) - verificar qual arquivo (dados do problema) está sendo importado no arquivo ==> InputData__ReadThisFile.py
(2) - alterar a faixa temperature_low e temperature_high (são os limites de busca)
(3) - o step é de quanto em quanto a função vai ser avaliada (sugestão ser <= 1 Kelvin)
'''

import logging
import numpy as np
from scipy.optimize import newton, fsolve, root, brentq, brenth, brent, brute, fmin
from Wilson import calculate_K_values_wilson
from EOS_PengRobinson import PengRobinsonEos
from Michelsen import Michelsen
from InputData___ReadThisFile import props

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

#TO LOGGING MSG(import logging and logging.basicConfig and logging.disable)
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
#logging.disable(logging.CRITICAL)


#FOR BUBBLE T
(pressure, _, global_molar_fraction,
        critical_pressure, critical_temperature, acentric_factor,
        molar_mass, omega_a, omega_b, binary_interaction, specific_heat) = props
#NECESSARY OBJECTS
michelsen_obj = Michelsen()
eos_obj = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor,
                              omega_a, omega_b, binary_interaction)

def function_inner_loop(temperature, pressure, global_molar_fraction, max_iter = 100, tolerance = 1.0e-12, stability=False):
    P = pressure
    T = temperature
    xi = z = global_molar_fraction
    f_L, Z_L = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, xi, 'liquid')
    PHI_L = f_L / (xi * P)
    initial_K_values = calculate_K_values_wilson(P, T, eos_obj.Pc, eos_obj.Tc, eos_obj.ω)
    is_stable, K_michelsen = michelsen_obj(eos_obj, P, T, z, initial_K_values, max_iter, tolerance)
    if stability:
        if not is_stable:
            msg = str('===> two-phases can be in equilibrium')
        else:
            msg = str('===> one phase in this @')
        print(str(msg))
    logging.debug('temperature' + str(T))
    logging.debug('K_michelsen' + str(K_michelsen))
    logging.debug('One-phase = ' + str(is_stable))
    yi = xi * K_michelsen
    Sy = np.sum(yi)
    Sy0 = 2 * Sy
    itermax = 2
    for counter in np.arange(itermax):
        if is_stable:
            break
        else:
            pass
        f_V, Z_V = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, yi, 'vapor')
        PHI_V = f_V / (yi * P)
        K = PHI_L / PHI_V
        yi = xi * K
        Sy = np.sum(yi)
        logging.debug('iteracao' + str(counter))
        logging.debug('K' + str(K))
        logging.debug('yi' + str(yi))
        logging.debug('Sy' + str(Sy) + '\n----')
        if (np.abs(Sy - Sy0) < tolerance):
            break
        else:
            yi /= Sy
            Sy0 = np.copy(Sy)
    return Sy, yi

#IT WAS NECESSARY APPLY NESTED FUNCTION BECAUSE SCIPY.OPTIMIZE DOES NOT ACCEPT TUPLE IN return OF OBJECTIVE FUNCTION
def function_outer_loop(temperature, pressure, global_molar_fraction):
    Sy, yi = function_inner_loop(temperature, pressure, global_molar_fraction)
    return np.abs(Sy - 1.0)

def executable(temperature_low, temperature_high, step, pressure, global_molar_fraction):
    rranges = (slice(temperature_low, temperature_high, step),)
    Tbubble = brute(function_outer_loop, rranges, args=(pressure, global_molar_fraction), finish=None)
    #Tbubble = fsolve(function_outer_loop, temperature_low, args=(pressure, global_molar_fraction))
    return Tbubble

#FOR BUBBLE T
#WE MUST SET TEMPERATURE_LOW, TEMPERATURE_HIGH VARIABLES AND STEP
temperature_low = 374.
temperature_high = 377.
step = 1. / 2
Tbubble = executable(temperature_low, temperature_high, step, pressure, global_molar_fraction)
Sy, yi = function_inner_loop(Tbubble, pressure, global_molar_fraction, stability=True)



if ((Tbubble / temperature_low) < 1.001):
    print('ATTENTION: the Tb can be out of LOW limit you have set')
elif ((Tbubble / temperature_high) > 0.98):
    print('ATTENTION: the Tb can be out of HIGH limit you have set')

print('\n---------------------------------------------------')
print('[1] - Seek the Tb [K] in --> [LL = %.3f & HL = %.3f]' % (temperature_low, temperature_high) + '\n')
print('[2] - >>> BUBBLE TEMPERATURE FOUND <<< = %.3f [K]' % Tbubble + '\n')
print('[4] - Concentration vapor fase = ', yi.round(3))
print('[5] - Pay attention if Sy is close to unity (Sy = %.6f)' % Sy)


