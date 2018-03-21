class Molecule:
    """
    Store molecule info here
    """
    def __init__(self, name, molar_mass, TC, pC, AcF, Cp):
        """
        Pass parameters desribing molecules
        """
        #! name
        self.name = name
        #! Molar mass (kg/kmol)
        self.MM = molar_mass
        #! Critical temperature (K)
        self.TC = TC
        #! Critical pressure (bar)
        self.pC = pC
        #! Specific heat
        self.Cp = Cp
        #! Accentric factor
        self.AcF = AcF

    def print_parameters(self):
        """
        Print molecule parameters.
        """
        print("""Molecule: %s.
        \tMM (molar mass) = %.3f
        \tTC = %.1f [K] -- TC = %.1f [C]
        \tpC = %.4e [Pa] -- pC = %.3f [bar]
        \tCp = %.3f [J / (kmol K)]
        \tAcF (acentric factor) = %f""" % (self.name, self.MM, self.TC, (self.TC - 273.15),
                                           self.pC, (self.pC / 1e5), self.Cp, self.AcF))