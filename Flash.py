import numpy as np

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)



class Flash: 
    def __init__(self, max_iter = 50, tolerance = 1.0e-13, print_statistics=False):
        self.max_iter, self.tolerance, self.print_statistics = max_iter, tolerance, print_statistics
    
    def flash_residual_function(self, x, T, P, eos_obj, z):
        size = x.shape[0]

        # Get values from unknown vector
        K = x[0:size - 1]  # K-values
        F_V = x[size - 1]

                
        LI = 0.0001
        LS = 0.9990
        F_V = np.select([LI < F_V < LS, F_V <= LI, F_V >= LS],[F_V, 0.0, 1.0])
        

        x_L = z / (F_V * (K - 1.0) + 1.0)
        x_V = K * x_L

        # Vapor
        f_V, Z_V = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, x_V, 'vapor')

        # Liquid
        f_L, Z_L = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, x_L, 'liquid')

        residual_fugacity = f_L - f_V
        residual_mass = np.sum(z * (K - 1.0) / (1.0 + F_V * (K - 1.0)))
        residual = np.r_[residual_fugacity, residual_mass]

        return residual


    def __call__(self, rr_obj, eos_obj, P, T, z, K_takecare): 
        max_iter = self.max_iter
        tolerance = self.tolerance
        print_statistics = self.print_statistics
        K = K_takecare
    

        # Initialize error with some value
        error = 100.0
        counter = 0
        while error > tolerance and counter < max_iter:
            F_V = rr_obj(z, K)                                  
            x_L = z / (F_V * (K - 1.0) + 1.0)
            x_V = K * x_L

            # Vapor
            f_V, Z_V = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, x_V, 'vapor')

            # Liquid
            f_L, Z_L = eos_obj.calculate_fugacities_with_minimum_gibbs_energy(P, T, x_L, 'liquid')

            f_ratio = f_L / f_V
            K *= f_ratio

            error = np.einsum('i,i', (f_ratio - 1.0), (f_ratio - 1.0))   #<---eq 4.30, pg 6                   
            counter += 1
            #print('yi:', x_V, 'xi:', x_L, 'K:', K, 'fvi:', f_V, 'fli', f_L,'fLi/fVi:', f_ratio, '\n' )

        if print_statistics:
            print('SS Flash: %d iterations, error is %g.' % (counter, error))

        return K, F_V

