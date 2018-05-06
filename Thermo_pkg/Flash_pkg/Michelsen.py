import numpy as np

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

class Michelsen:
    def __init__(self):
        pass
    
    def ss_stability_test(self, eos_obj, P, T, z, test_type, K_values_initial, max_iter, tolerance):
        
        K = np.copy(K_values_initial)

        error = 100.0

        f_ref, Z_ref = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, z, test_type)

        counter = 0
        while error > tolerance and counter < max_iter:
            if test_type is 'vapor':
                other_type = 'liquid'
                x_u = z * K
            else:
                assert test_type is 'liquid', 'Non existing test_type! ' + test_type
                other_type = 'vapor'
                x_u = z / K

            sum_x_u = np.einsum('i->', x_u) 
            x_u_normalized = x_u / sum_x_u

            f_u, Z_u = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, x_u_normalized, test_type)

            if test_type is 'vapor':
                correction = f_ref / (f_u * sum_x_u)
            else:
                assert test_type is 'liquid', 'Non existing test_type! ' + test_type
                correction = (f_u * sum_x_u) / f_ref

            K *= correction
            error =  np.einsum('i,i', (correction - 1.0), (correction - 1.0))   #<----- eq 4.69, pg 13
            counter += 1
            

        return sum_x_u, K

    def __call__(self, eos_obj, P, T, z, K_values_initial, max_iter=100, tolerance=1.0e-12): 
        
        sum_vapor, K_values_vapor = self.ss_stability_test(eos_obj, P, T, z, 'vapor', K_values_initial, max_iter, tolerance)
        sum_liquid, K_values_liquid = self.ss_stability_test(eos_obj, P, T, z, 'liquid', K_values_initial, max_iter, tolerance)

        sum_ln_K_vapor =  np.einsum('i,i', np.log(K_values_vapor), np.log(K_values_vapor)) 
        sum_ln_K_liquid = np.einsum('i,i', np.log(K_values_liquid), np.log(K_values_liquid)) 
        crit = 1.0e-4 #Criterion applied to decide if trivial solution is being approached (eq, 4.71, pg 13)  
        
        if (sum_ln_K_vapor - crit < 0.0 ):
            TSV = True                    #<---Trivial Solution Vapor
        else:
            TSV = False
        
        if (sum_ln_K_liquid - crit < 0.0 ):
            TSL = True                   #<---Trivial Solution Liquid
        else:
            TSL = False 
        
        SV = (sum_vapor - 1.0)
        SL = (sum_liquid - 1.0)
        tol = 1.0e-8

       # Table 4.6 from Phase Behavior
        if TSV and TSL:
            is_stable = True
        elif SV <= tol and TSL:
            is_stable = True
        elif TSV and SL <= tol:
            is_stable = True
        elif SV <= tol and SL <= tol:
            is_stable = True
        elif SV > tol and TSL:
            is_stable = False 
        elif TSV and SL > tol:
            is_stable = False
        elif SV > tol and SL > tol:
            is_stable = False
        elif SV > tol and SL <= tol:
            is_stable = False
        elif SV <= tol and SL > tol:
            is_stable = False
        else:
            assert False, 'ERROR: No stability condition found...'

        if not is_stable:
            K_values_estimates = K_values_vapor * K_values_liquid
        else:
            K_values_estimates = np.copy(K_values_initial)
            
        

        return is_stable, K_values_estimates
    
    

