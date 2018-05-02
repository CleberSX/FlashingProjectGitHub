import numpy as np

# D = 16e-3
# dth = 4e-3

# AR = (dth / D) ** 2
# Gt = 609.5

# Dp = 58e3

# densL = (Gt ** 2) * (1. - AR ** 2) / (2 * Dp)

'''Venturi''' 

Gt = 609.5
Dp = 58e3
Kventuri = 0.98

densL = (1. / (2 * Dp)) * (Gt / Kventuri) ** 2

print(densL)
