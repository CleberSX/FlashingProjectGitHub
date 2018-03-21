from Molecule import Molecule
import numpy as np
import Tools_Convert

class InputData_class:
    """
    Store molecule info here
    """
    def __init__(self, p, T, MM, LC, base):
        self.p, self.T, self.MM, self.LC, self.base = p, T, MM, LC, base


    def __call__(self):
        zin = np.array([self.LC, (1. - self.LC)])
        z, z_mass = Tools_Convert.frac_input(self.MM, zin, self.base)
        return (self.p, self.T, z, z_mass)

    def print_input_data(self):
        """
        Print the input data
        """
        p, T, z, z_mass = self.__call__()
        print('\tPressure: %.3e [Pa]' % p)
        print('\tTemperature = %.3f [K]' % T)
        print('\tGlobal concentration [molar] ', z.round(3))
        print('\tGlobal concentration [mass] ', z_mass.round(3))


