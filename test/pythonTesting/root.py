import scipy
import math

def f(k):
    return math.exp(k) - 1

def derF(k):
    return math.exp(k)

print(scipy.optimize.newton(func=f, x0=2, fprime=derF))

