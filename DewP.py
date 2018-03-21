'''
ESTA FUNCAO CALCULA A PRESSAO DE ORVALHO EM Pa.
PONTOS QUE DEVEMOS ALIMENTAR/ALTERAR:
(1) - verificar qual arquivo (dados do problema) está sendo importado no arquivo ==> InputData__ReadThisFile.py
(2) - alterar a faixa pressure_low e pressure_high (sao os limites de busca)
(3) - o step eh de quanto em quanto a funcao serah avaliada (sugestao: 5000 Pa)
'''

import logging
import numpy as np
from scipy.optimize import newton, fsolve, root, brentq, brenth, brent, brute
from Wilson import calculate_K_values_wilson
from EOS_PengRobinson import PengRobinsonEos
from Michelsen import Michelsen
from InputData___ReadThisFile import props

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

#TO LOGGING MSG(import logging and logging.basicConfig and logging.disable)
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
#logging.disable(logging.CRITICAL)


#PROPS IS A LIST OF PROPERTIES; BLANK SPACE CORRESPOND TO THE PROPERTY WE ARE INTERESTED
(_, temperature, global_molar_fraction,
critical_pressure, critical_temperature, acentric_factor,
molar_mass, omega_a, omega_b, binary_interaction, specific_heat) = props


#NECESSARY OBJECTS
eos_obj = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor,
                              omega_a, omega_b, binary_interaction)
michelsen_obj = Michelsen()


def calculate_pressure_guess(global_molar_fraction, temperature):
    '''This function give a initial guess to dew pressure. Based on its result, a range of pressure
    is then created. The numerical method will seek the correct dew pressure considering this range'''
    z = global_molar_fraction
    T = temperature
    Pc = eos_obj.Pc
    Tc = eos_obj.Tc
    ω = eos_obj.ω
    argum = ( Pc * np.exp(5.37 * (1 + ω) * (1 - (Tc / T))))
    num_den = z / argum
    soma = np.sum(num_den)
    logging.debug('Pc * np.exp(5.37 * (1 + ω) * (1 - (Tc / T)))' + str(argum))
    logging.debug('z / argum' + str(num_den))
    logging.debug('soma' + str(soma))
    return np.reciprocal( soma )


def function_inner_loop(pressure, temperature, global_molar_fraction, max_iter = 100,
                        tolerance = 1.0e-12, stability=False):
    T = temperature
    P = pressure
    logging.debug('pressure' + str(P))
    yi = z = global_molar_fraction
    f_V, Z_V = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, yi, 'vapor')
    PHI_V = f_V / (yi * P)
    initial_K_values = calculate_K_values_wilson(P, T, eos_obj.Pc, eos_obj.Tc, eos_obj.ω)
    is_stable, K_michelsen = michelsen_obj(eos_obj, P, T, z, initial_K_values, max_iter, tolerance)
    if stability:
        if not is_stable:
            msg = str('===> two-phases can be in equilibrium')
        else:
            msg = str('===> one phase in this @')
        print(msg)
    logging.debug('K_michelsen' + str(K_michelsen))
    logging.debug('One-phase = ' + str(is_stable))
    xi = yi #yi / K_michelsen
    Sx = np.sum(xi)
    Sx0 = 2 * Sx
    itermax = 2
    for counter in np.arange(itermax):
        if is_stable:
            Sx = (Sx + 3.0)  # To force the program stores wrong Sx's value [avoid the solver seeks minimum values of (Sx - 1)]
            break
        else:
            pass
        f_L, Z_L = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, xi, 'liquid')
        PHI_L = f_L / (xi * P)
        K = PHI_L / PHI_V
        xi = yi / K
        Sx = np.sum(xi)
        logging.debug('iteracao' + str(counter))
        logging.debug('K' + str(K))
        logging.debug('xi' + str(xi))
        logging.debug('Sx = ' + str(Sx) + '\n----')
        if (np.abs(Sx - Sx0) < tolerance):
            break
        else:
            xi /= Sx
            Sx0 = np.copy(Sx)
    return Sx, xi

def function_outer_loop(temperature, pressure, global_molar_fraction):
    Sx, xi = function_inner_loop(temperature, pressure, global_molar_fraction)
    return np.abs(Sx - 1.0)


def executable(pressure_low, pressure_high, step, temperature, global_molar_fraction):
    ranges = (slice(pressure_low, pressure_high, step),)  # Go from Start to End applying this step size...
                                                          # ... (step length instead quantity of points)
    Pdew = brute(function_outer_loop, ranges, args=(temperature, global_molar_fraction), finish=None)
    #Pdew = fsolve(function_outer_loop, pressure_low, args=(temperature, global_molar_fraction))
    Sx, xi = function_inner_loop(Pdew, temperature, global_molar_fraction, stability=True)
    return Pdew, Sx, xi


pguess = calculate_pressure_guess(global_molar_fraction, temperature)
pressure_high = 200e3#* pguess
pressure_low = 30e3 # * pguess
step = 10

Pdew, Sx, xi = executable(pressure_low, pressure_high, step, temperature, global_molar_fraction)




if ((Pdew / pressure_low) < 1.001):
    print('ATTENTION: the Pd can be out of LOW limit you have set')
elif ((Pdew / pressure_high) > 0.98):
    print('ATTENTION: the Pd can be out of HIGH limit you have set')


print('\n---------------------------------------------------')
print('[1] - Guess Pd [Pa] = %.3e' % pguess)
print('[2] - Seek the Pd in [Pa]--> [LL = %.3e & HL = %.3e]' % (pressure_low, pressure_high) + '\n')
print('[3] - >>> DEW PRESSURE FOUND <<< = %.3e [Pa]' % Pdew + '\n')
print('[4] - Concentration liquid fase = ', xi.round(3))
print('[5] - Pay attention if Sx is close to unity (Sx = %.6f)' % Sx)








