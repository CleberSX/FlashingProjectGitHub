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

