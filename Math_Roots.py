
from math import sqrt, acos, cos
import numpy as np
R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

class FindZ:
    def __init__(self):
            pass
        
    def cbrt(self, arg):
        if arg >= 0.0:                
            return arg ** (1.0/3.0)
        else:
            return -(abs(arg) ** (1.0/3.0))
    
    @staticmethod
    def calculate_roots_of_cubic_eos(p0, p1, p2, p3):
        coef_a = (3.0 * p2 - (p1 ** 2)) / 3.0        
        coef_b = (2.0 * (p1 ** 3) - 9.0 * p1 * p2 + 27.0 * p3) / 27.0        
        delta = 0.25 * (coef_b ** 2) + (coef_a ** 3) / 27.0     
        obj_findz = FindZ()

        roots = []
        if delta > 0.0:
            # 1 real root, 2 imaginary                 
            const_A =  obj_findz.cbrt(-0.5 * coef_b + sqrt(delta))
            const_B =  obj_findz.cbrt(-0.5 * coef_b - sqrt(delta))
            aux1 = (const_A + const_B)
            single_root = aux1 - p1 / 3.0
            try:
                roots.append(single_root)
                if single_root < 0.0:
                    raise Exception('The EOS has only one root and this is negative = %f' % single_root)
            except Exception as my_msg_single:
                print('Ocorreu um erro (raiz negativa) no Math_Roots.py: ' + str(my_msg_single))

        else:
            # 3 real roots
            phi = acos(-0.5 * coef_b / sqrt(-(coef_a ** 3) / 27.0))
            root_1 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0) - p1 / 3.0
            root_2 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0 + 2.0 * np.pi / 3.0) - p1 / 3.0
            root_3 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0 + 4.0 * np.pi / 3.0) - p1 / 3.0

            smallest_root = min(min(root_1,root_2), root_3)
            largest_root = max(max(root_1, root_2), root_3)

            try:
                roots.append(smallest_root)
                if smallest_root < 0.0:
                    raise Exception('The EOS has 3 roots. Probably the smallest is negative = %f' % smallest_root)
            except Exception as my_msg_two:
                print('Ocorreu um erro (raiz negativa!): ' + str(my_msg_two))

            roots.append(largest_root)
        return roots

