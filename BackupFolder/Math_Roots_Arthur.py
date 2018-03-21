
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

            assert single_root > 0.0, 'Z-factor < 0.0! Delta is %f, %f' % (delta, single_root)

            roots.append(single_root)


        else:
            # 3 real roots
            phi = acos(-0.5 * coef_b / sqrt(-(coef_a ** 3) / 27.0))
            root_1 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0) - p1 / 3.0
            root_2 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0 + 2.0 * np.pi / 3.0) - p1 / 3.0
            root_3 = 2.0 * sqrt(-coef_a / 3.0) * cos(phi / 3.0 + 4.0 * np.pi / 3.0) - p1 / 3.0


            smallest_root = min(min(root_1,root_2), root_3)
            assert smallest_root > 0.0, 'Z-factor < 0.0! Delta is %f, %f' % (delta, smallest_root)

            largest_root = max(max(root_1,root_2), root_3)
            assert largest_root > 0.0, 'Z-factor < 0.0! Delta is %f, %f' % (delta, largest_root)


            roots.append(smallest_root)
            roots.append(largest_root)

        return roots

