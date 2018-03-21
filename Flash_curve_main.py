import logging
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from Wilson import calculate_K_values_wilson
from RachfordRice import RachfordRice
from Flash import Flash
from EOS_PengRobinson import PengRobinsonEos
from Michelsen import Michelsen
from Properties import Properties
from InputData___ReadThisFile import props #Check this InputData__ReadThisFile.py to see the problem data

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)


#TO LOGGING MSG(import logging and logging.basicConfig and logging.disable)
logging.basicConfig(level=logging.DEBUG, format=' %(asctime)s - %(levelname)s - %(message)s')
#logging.disable(logging.CRITICAL)




def calculate_vapor_liquid_equilibrium(eos_obj, michelsen_obj, flash_obj, prop_obj, pressure, temperature, 
                                       molar_mass, global_molar_fractions, max_iter, tolerance,
                                       print_statistics=None):
    P = pressure
    T = temperature
    z = global_molar_fractions
    Mi = molar_mass
    
    #size = z.shape[0]

    # Estimate initial K-values
    initial_K_values = calculate_K_values_wilson(pressure, temperature, eos_obj.Pc, eos_obj.Tc, eos_obj.ω)
    
    # Check if the mixture is stable and takes Michelsen's K
    is_stable, K_michelsen = michelsen_obj(eos_obj, P, T, z, initial_K_values, max_iter, tolerance)
    
    # Executing the Flash
    K_flash, F_V_flash = flash_obj(rr_obj, eos_obj, pressure, temperature, global_molar_fractions, K_michelsen)
    
    # Good estimate!
    Vector_ToBe_Optimized = np.append(K_flash, F_V_flash)
    
    
   
    ## Use estimates from flash (successive substitutions)!!!
    if 0.0 <= F_V_flash <= 1.0:
        result, infodict, ier, mesg = fsolve(func=flash_obj.flash_residual_function, x0=Vector_ToBe_Optimized, 
                                             args=(T, P, eos_obj, z), full_output=True) 
             
        size = result.shape[0]
        K_values_newton = result[0:size-1]
        F_V = result[size-1]
        x_i = global_molar_fractions / (F_V * (K_values_newton -1) + 1)
        y_i = K_values_newton * x_i
        f_L, Z_L = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, x_i, 'liquid')
        f_V, Z_V = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, y_i, 'vapor')
        rho_L = prop_obj.calculate_density_phase(P, T, Mi, x_i, 'liquid')
        rho_V = prop_obj.calculate_density_phase(P, T, Mi, y_i, 'vapor')
        if print_statistics:
            print('Newton flash converged? %d, %s' % (ier, mesg))

    elif F_V_flash < 0.0:
        x_i = z
        y_i = np.zeros_like(z) 
        K_values_newton = initial_K_values
        rho_L = prop_obj.calculate_densite_phase(P, T, Mi, x_i, 'liquid')
        rho_V = 0.0
        F_V = 0.0
    elif F_V_flash > 1.0:
        y_i = z
        x_i = np.zeros_like(z) 
        K_values_newton = initial_K_values
        rho_L = 0.0
        rho_V = prop_obj.calculate_densite_phase(P, T, Mi, y_i, 'vapor')
        F_V = 1.0
    else:
        raise Exception("Nenhuma das @ foram dectadas na calculate_vapor_liquid_equilibrium")
        
    
    
    return F_V, K_values_newton, x_i, y_i, rho_V, rho_L

def calculate_molar_fraction_curve(pressure, temperature_init, temperature_max, molar_mass, global_molar_fractions,
                                   max_iter=100, tolerance=1.0e-12, print_statistics=None):
    P = pressure
    z = global_molar_fractions
    Mi = molar_mass
    size = z.shape[0]
    temperature_points = np.linspace(int(temperature_init), int(temperature_max), 7)
    LN = temperature_points.shape[0]  #lines' number
    CN = 2*size + 3  #columm's number (a mixture with 3 components has: [x1,x2,x3], [y1,y2,y3], F_V, rho_V and rho_L)
    matrix_results = np.zeros((LN, CN))

    for index, T in enumerate(temperature_points):
        F_V, K, x_i, y_i, rho_V, rho_L = calculate_vapor_liquid_equilibrium(eos_obj, michelsen_obj, flash_obj,
                                                                                    prop_obj, P, T, Mi, z,  max_iter,
                                                                                    tolerance, print_statistics)
        #print('Temp', T, 'F_V', F_V, 'x1, x2', x_i, 'y1,y2', y_i, 'rho_V', rho_V, 'rho_L', rho_L)
        matrix_results[index,:] = np.r_[F_V, x_i, y_i, rho_V, rho_L]

    if print_statistics:
        print('Temperature: %g K' % T)
    return  matrix_results




# [1] - Allocating the variables, with P and T places left empty
(_, _, global_molar_fractions, 
critical_pressure, critical_temperature, acentric_factor,
molar_mass, omega_a, omega_b, binary_interaction, specific_heat) = props
# [2] - Finally, specifying the pressure and temperature to the problem
pressure = 1.01325 * 1e5 # [bar to Pa]
Tinit = 335.0
Tmax = 355.0

# [3] - CREATING OBJECTS
#eosroots_obj = roots_and_choosing_roots_cubic_equation()
rr_obj = RachfordRice()
eos_obj = PengRobinsonEos(critical_pressure, critical_temperature, acentric_factor, 
                      omega_a, omega_b, binary_interaction)
michelsen_obj = Michelsen()
prop_obj = Properties(critical_pressure, critical_temperature, acentric_factor, 
                      omega_a, omega_b, binary_interaction)

flash_obj = Flash(max_iter = 50, tolerance = 1.0e-13, print_statistics=None)


# [4] - Calling the main function, which one responsible by calculate the quality and molar fractions
matrix_results = calculate_molar_fraction_curve(pressure, Tinit, Tmax, molar_mass, global_molar_fractions,
                                                       max_iter=100, tolerance=1.0e-12, print_statistics=None)



# [5] - Matrix result
cn = global_molar_fractions.shape[0] #column's number
# Finally...
quality = matrix_results[:,0]
x_i = matrix_results[:,1:cn+1]
y_i = matrix_results[:,cn+1:2*cn+1]
rho_V = matrix_results[:,-2]
rho_L = matrix_results[:,-1]
temperature_points = np.linspace(int(Tinit), int(Tmax), 7)


logging.debug('The list of temperature is = ' + str(temperature_points) + '\n------')
logging.debug('vapor quality while temperature increase are = \n--' + str(quality) + '\n------')
logging.debug('x_i is = \n--' + str(x_i) + '\n------')
logging.debug('y_i is = \n--' + str(y_i))
'''


#Bubble-Point-Temperature: Look for approx. zero in F_V; then look for corresponding element in pressure_points vector
#the [0] at the end is for taking just the first element
BPT = temperature_points.ravel()[np.flatnonzero(quality <= 1.0e-3)][0]
print("A temperature de bolha é de: %0.1f K para uma pressão de %0.1f bar" %(BPT, pressure)) '''


# [6] - Plotting
plt.plot(temperature_points, quality, label='Vapor Quality')
plt.xlabel('Temperature [K]')
plt.ylabel('Quality molar [mol/mol]')
plt.legend(loc='upper center')
plt.axis([np.min(temperature_points), np.max(temperature_points), 0.0, 1.0,])
plt.show()
plt.close()


componentes=['C1', 'C4', 'C10']
cores=['r','g','b']
phase = ['vapor mixture', 'liquid mixture']


#plt.plot(x_i[:, 0], temperature_points, label=componentes[0], color = cores[0])
for i in np.arange(cn):
    plt.plot(temperature_points, x_i[:,i], label=componentes[i], color = cores[i])
plt.xlabel('Temperature [K]')
plt.ylabel('Liquid composition [mol/mol]')
plt.legend(loc='upper center')
plt.axis([np.min(temperature_points), np.max(temperature_points), 0.0, 1.0])
plt.show()
plt.close()


for i in np.arange(cn):
    plt.plot(temperature_points, y_i[:,i], label=componentes[i], color = cores[i])
plt.xlabel('Temperature [K]')
plt.ylabel('Vapor composition [mol/mol]')
plt.legend(loc='upper center')
plt.axis([np.min(temperature_points), np.max(temperature_points), 0.0, 1.0])
plt.show()
plt.close()


# plt.plot(temperature_points, rho_V, label=phase[0], color = cores[0])
# plt.plot(temperature_points, rho_L, label=phase[1], color = cores[2])
# plt.xlabel('Temperature [K]')
# plt.ylabel('Mixture density [kg/m3]')
# plt.legend(loc='upper right')
# plt.axis([np.min(temperature_points), np.max(temperature_points), np.min(rho_V), 1.1*(np.max(rho_L))])
# plt.show()
# plt.close()

