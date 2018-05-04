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


'''
=========================
BRINCANDO COM CLASS
========================
 '''

class MyParentClass():
    def __init__(self, x, y):
        self.x, self.y = x, y

    def soma(self):
        return self.x + self.y





class SubClass(MyParentClass):
    def __init__(self, x, y, z, w, p):
        self.z, self.w, self.p = z, w, p
        super().__init__(x, y)
    

        
    def somaElevadop(self, K):
        w, z, p = self.w, self.z, self.p
        return super().soma() ** p * K + (z + w)

# obj_parent = MyParentClass(0.001, 0.232)
obj = SubClass(10., 20., 1., 2., 3.)



print(obj.somaElevadop(-1.))


'''
=========================
BRINCANDO COM WRAP FUNCTIONS
========================
 '''

def fexterna(func):

 
   def finterna(*args, **kwargs):
       print("Chamando funcao: %s()"  % (func.__name__))
       result = func(*args, **kwargs) + 324.12
       return result
 
   return finterna
 
def dobro(x):
    """ Uma funcao exemplo qualquer.
    """
    return 2*x
 
dobro_com_print = fexterna(dobro)
print(dobro_com_print(10))


'''ou simplesmente faça com uso de decorador (Python traz essa facilidade)'''
# @fexterna
# def dobro(x):
#     """ Uma funcao exemplo qualquer.
#     """
#     return 2*x
'''para simplesmente chamar:'''
# print(dobro(5))

'''
=====================
USANDO SYMPY
=================
'''
from sympy import Function
from sympy import lambdify, Matrix
from sympy.utilities.lambdify import implemented_function
# from sympy import sqrt, sin, Matrix
from sympy.abc import w, x, y, z

g = lambdify(y, y**3)
print(g(2))

'''
=====================
USANDO DERIVADAS COM NUMPY
=================
'''
p = np.poly1d([1, 0, 1])
print(p)
q = p.deriv()
print(q)

