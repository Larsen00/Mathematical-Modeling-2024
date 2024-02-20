import sympy as s
from sympy.plotting import plot


f = lambda x, mu : s.exp(-(x-mu)**2/(2*6.7**2))/(6.7*s.sqrt(2*s.pi))
    



p1 = plot(f(s.symbols('x'), 175.5), show=False)
p2 = plot(f(s.symbols('x'), 175.5), show=False)
p1.show()