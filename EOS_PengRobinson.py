import numpy as np
from Math_Roots import FindZ

R = 8314  # [J/(kmol*K)]
SQRT_2 = np.sqrt(2)

class PengRobinsonEos:
    def __init__(self, Pc, Tc, ω, Ω_a, Ω_b, κ_ij):
        self.Pc   = Pc
        self.Tc   = Tc
        self.ω    = ω
        self.Ω_a  = Ω_a
        self.Ω_b  = Ω_b
        self.κ_ij = κ_ij

    def α_function(self, T):
        ω, Tc = self.ω, self.Tc
        self.m = np.where(
            ω < 0.49,
            0.374640 + 1.54226 * ω - 0.269920 * (ω ** 2),
            0.379642 + 1.48503 * ω - 0.164423 * (ω ** 2) + 0.016667 * (ω ** 3)
        )
        return (1.0 + self.m * (1.0 - np.sqrt(T / Tc))) ** 2

    
    def update_eos_coefficients(self, P, T, x):
        κ_ij = self.κ_ij
        Pc = self.Pc   
        Tc = self.Tc 
        Ω_a = self.Ω_a 
        Ω_b = self.Ω_b 
        
        α = self.α_function(T)

        self.a_pure = (Ω_a * α * (R * Tc) ** 2) / Pc
        self.b_pure = (Ω_b * R * Tc) / Pc
        self.A_pure = self.a_pure * P / (R * T) ** 2
        self.B_pure = self.b_pure * P / (R * T)
        
        A_ij = (1.0 - κ_ij) * np.sqrt(np.einsum('i,j', self.A_pure, self.A_pure)) #(m,n)
        
        # This variables will be used in the fugacity expression
        self.x_j_A_ij = np.einsum('j,ij', x, A_ij)       #(m,) vector
        self.A_mix = np.einsum('i,i', x, self.x_j_A_ij)  #scalar
        self.B_mix = np.einsum('i,i', x, self.B_pure)    #scalar
        
      
    def calculate_eos_roots(self, fluid_type):
        A_mix = self.A_mix
        B_mix = self.B_mix

        p0 = 1.0
        p1 = - (1.0 - B_mix)
        p2 = A_mix - 3.0 * (B_mix ** 2) - 2.0 * B_mix
        p3 = -(A_mix * B_mix - B_mix ** 2 - B_mix ** 3)
        
        return FindZ.calculate_roots_of_cubic_eos(p0, p1, p2, p3)

        
    def calculate_fugacities(self, P, T, Z, x):

        part1 = (self.B_pure / self.B_mix) * (Z - 1.0) - np.log(Z - self.B_mix)
        part2 = (self.A_mix / (2.0 * SQRT_2 * self.B_mix))
        part3 = ((self.B_pure / self.B_mix) - 2.0 * self.x_j_A_ij / self.A_mix)
        part4 = np.log((Z + (1.0 + SQRT_2) * self.B_mix) / (Z + (1.0 - SQRT_2) * self.B_mix))

        PHI = np.exp(part1 + part2 * part3 * part4)

        return (x * P) * PHI  # [Pa]

        
    def calculate_normalized_gibbs_energy(self, f, x):
        g = (x * np.log(f)).sum()
        return g
    
    def calculate_fugacities_with_minimum_gibbs_energy(self, P, T, x, fluid_type):
        # calculating all roots and if it has two possible roots
        # calculate both minimim gibbs energy and choose the 
        # group of fugacities with minimum gibbs energy

        self.update_eos_coefficients(P, T, x) #Without this line the entire program fails

        Z = self.calculate_eos_roots(fluid_type)
         
        if len(Z) == 1:
            f0 = self.calculate_fugacities(P, T, Z[0], x)
            return f0, Z[0]
        else:
            f0 = self.calculate_fugacities(P, T, Z[0], x)
            f1 = self.calculate_fugacities(P, T, Z[1], x)
            g0 = self.calculate_normalized_gibbs_energy(f0, x)
            g1 = self.calculate_normalized_gibbs_energy(f1, x)
            return np.select([g0 < g1, g0 > g1],[[f0, Z[0]], [f1, Z[1]]])
        

