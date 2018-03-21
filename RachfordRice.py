import numpy as np
from scipy.optimize import brentq, bisect, fsolve

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)


class RachfordRice:
    def __init__(self, F_Vguess = 0.5):
        self.F_Vguess = F_Vguess
        
        #parameter ci - see pg 7 - equation 4.39
    def calculate_auxiliar(self, K_values):
        self.F_Vmin = 0.999 / (1.0 - np.min(K_values))
        self.F_Vmax = 0.999 / (1.0 - np.max(K_values))
        
    @staticmethod
    def function_to_be_solve(F_Vguess, z, K_values):
        try:
            c = ( 1.0 / (K_values - 1.0) )
        except ZeroDivisionError as MyPersonalMsg:
            print("Divisão por zero no cálculo do parâmetro 'C' em "
                  "function_to_be_solve na class RachfordRice", MyPersonalMsg)
            
        return np.sum(z / (c + F_Vguess))
            
        
    def __call__(self, z, K_values):
        self.calculate_auxiliar(K_values)
        F_Vmin, F_Vmax = self.F_Vmin, self.F_Vmax

        
        try:
            F_V, converged = brentq(RachfordRice.function_to_be_solve, F_Vmin, F_Vmax, args=(z, K_values),  xtol=1e-16,
                                    full_output=True, disp=True)
            if converged is False:
                raise Exception('Fora da regiao bifasica ' + str(converged))
        except Exception as my_personal_error:
            print('Algum erro genérico no método da Class RachfordRice: vc precisa identificar ' + str(my_personal_error))
        return F_V

