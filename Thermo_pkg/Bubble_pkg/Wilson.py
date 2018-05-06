import numpy as np


def calculate_K_values_wilson(p, T, pC, Tc, ω):
    return (pC / p) * np.exp(5.37 * (1 + ω) * (1 - (Tc / T)))

