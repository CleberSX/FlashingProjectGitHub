
import logging
import numpy as np
import matplotlib.pyplot as plt
from EOS_PengRobinson import PengRobinsonEos
import Tools_Convert
from BubbleP import Bubble_class
from InputData___ReadThisFile import props, exp_data, comp1, comp2
R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)


'''
=================
  WHEN THE OBJECTIVE IS CREATE A ISOTHERM CURVE: this code is important to plot: [refrigerant] vs Pb ...
                                                ...for a specific isotherm                                                                                                
LEGEND:
---------
refrigerant_global_mass_fraction_points: it is the mixture [concentration] in mass of refrigerant
LN: number of lines 
CN: used to build columns, so we have decided to call number of columns 
global_mass_fractions_store: it is a vector created to store the global mass fraction [refrigerant, oil]
global_molar_fractions_store: it is a vector created to store the global molar fraction [refrigerant, oil]
Pbubble and Pbubble_store: the first is the bubble pressure @(zi, T); the last one stocks the former's values ...
                           ... considering a lot of values of zi, i.e., @({z1, z2, z3...}, T)
yi and yi_store: both represent the vapor phase concentration; the last one stocks the former's values...
                 ... considering a lot of values of zi, i.e., @({z1, z2, z3...}, T)
Sy and Sy_store: both represent the vapor phase concentration sum; the last one stocks the former's values...
                 ... considering a lot of values of zi, i.e., @({z1, z2, z3...}, T)

Obs:
--------
This code below determines the bubble pressure in function of refrigerant mass fraction and temperature: Pb = Pb(zR,T)
So, the list "props", which wraps up all imported variables, must have the places used by variables ...
...pressure, temperature and global_molar_fraction left empties (-,-,-, critical pressure,...) = props 
=================
'''



'''
=======
IMPORTING MOLECULAR & EXPERIMENTAL DATA (when you have experimental data) 
=======
'''

(pC, Tc, AcF, MM, omega_a, omega_b, kij, Cp) = props

(refrigerant_global_mass_fraction_expData_10Celsius, bubble_pressure_10Celsius,
            refrigerant_global_mass_fraction_expData_20Celsius, bubble_pressure_20Celsius,
            refrigerant_global_mass_fraction_expData_30Celsius, bubble_pressure_30Celsius,
            refrigerant_global_mass_fraction_expData_40Celsius, bubble_pressure_40Celsius,
            refrigerant_global_mass_fraction_expData_50Celsius, bubble_pressure_50Celsius,
            refrigerant_global_mass_fraction_expData_60Celsius, bubble_pressure_60Celsius,
            ) = exp_data


'''
=================================================================================================================
NECESSARY OBJECTS
=================================================================================================================
'''
eos_obj = PengRobinsonEos(pC, Tc, AcF, omega_a, omega_b, kij)
bubble_obj = Bubble_class(pC, Tc, AcF, kij)


'''
===================================
        Building the several isotherms and the complete molar concentration range to feed the code. Why does it necessary?

        Because Pb = Pb @(T, z): We want Pb for a specific global molar fraction and isotherm. But, in this case, we...
                                 ... are interesting a complete curve of bubble pressure to plot Pb x zi ... 
                                 ... for several isotherms 
===================================
'''

T_list = np.array([10.7, 20.2, 30.2, 40.7, 50.6, 60.], float) + 273.15
CN = T_list.shape[0]
LC_mass_list = np.linspace(0.01, 0.99, 11) #Lighter component list
LN = LC_mass_list.shape[0]
CompN = MM.shape[0] #components number
Pbubble_store = np.zeros([LN, CN]) #matrix with LN lines and CN columns
# Sy_store = np.zeros([LN, CN])
# y_store = np.zeros([LN, 2 * CN])



'''
========================
    PUTTING TO RUN: with a for loop
========================
'''

for index_out, T in enumerate(T_list):
    for index, LC_mass in enumerate(LC_mass_list):
        logging.debug('LC mass concentration =======> ' + str(LC_mass))
        logging.debug('Temperatura =======> ' + str(T))
        z_mass = np.array([LC_mass, (1. - LC_mass)])
        z = Tools_Convert.convert_massfrac_TO_molarfrac(MM, z_mass)
        pBubble, y, Sy, counter = bubble_obj(T, z)
        Pbubble_store[index, index_out] = pBubble
        # Sy_store[index, index_out] = Sy
        # y_store[index, 2 * index_out:2 * index_out + 2] = y
        # print('%.2f \t %.2e \t %.5f' % (T, pBubble, Sy))



'''
=====================
        PLOTTING
=====================
'''
colors = ['r', 'b', 'k', 'y', 'm', 'c', 'g', 'lightgray']
markers = ['o', 'v', '<', '>', '.', 'p', 'P', 's', '*']


plt.title('Bubble Pressure of Mixture of ' + comp1.name + comp2.name + '(Dados Artigo Moisés)')
for i in np.arange(CN):
    plt.plot(LC_mass_list, Pbubble_store[:, i],
             color = colors[i], marker = '', lw = 1.5,
             ls = '-', label = 'T = %.2f [C]' % (T_list[i] - 273.15)) #ls: lineshape
plt.plot(refrigerant_global_mass_fraction_expData_10Celsius, bubble_pressure_10Celsius, color = colors[0],
         marker = markers[0], ls='', label='Exp. Data 10.7 Celsius')
plt.plot(refrigerant_global_mass_fraction_expData_20Celsius, bubble_pressure_20Celsius, color = colors[1],
         marker = markers[1], ls='', label='Exp. Data 20.2 Celsius')
plt.plot(refrigerant_global_mass_fraction_expData_30Celsius, bubble_pressure_30Celsius, color = colors[2],
         marker = markers[2], ls='', label='Exp. Data 30.2 Celsius')
plt.plot(refrigerant_global_mass_fraction_expData_40Celsius, bubble_pressure_40Celsius, color = colors[3],
         marker = markers[3], ls='', label='Exp. Data 40.7 Celsius')
plt.plot(refrigerant_global_mass_fraction_expData_50Celsius, bubble_pressure_50Celsius, color = colors[4],
         marker = markers[4], ls='', label='Exp. Data 50.6 Celsius')
plt.plot(refrigerant_global_mass_fraction_expData_60Celsius, bubble_pressure_60Celsius, color = colors[5],
         marker = markers[5], ls='', label='Exp. Data 60.0 Celsius')
plt.xlabel('[lighter component (kg kg$^{-1}$)]')
plt.ylabel('Bubble Pressure [Pa]')
plt.legend(loc = 'upper left')
# Turn on the minor TICKS, which are required for the minor GRID
plt.minorticks_on()
plt.grid(which='major', linestyle='-', linewidth='0.5', color=colors[2])
plt.grid(which='minor', linestyle=':', linewidth='0.5', color=colors[-1])
plt.axis([0.0, 1.0, np.min(0.9 * Pbubble_store), np.max(1.1 * Pbubble_store)])
plt.show()
plt.close()


