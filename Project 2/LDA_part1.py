import sympy as s
from sympy.plotting import plot


'''
# SET PATH FOR OSX USERS
if sys.platform == 'darwin':
    sys.path.append('../code')
    
LOAD_MAIN_FLAG = False
if LOAD_MAIN_FLAG:
    from main import test_an_expression
'''

f = lambda x, mu : s.exp(-(x-mu)**2/(2*6.7**2))/(6.7*s.sqrt(2*s.pi))
    



p1 = plot(f(s.symbols('x'), 175.5), show=False)
p2 = plot(f(s.symbols('x'), 162.9), show=False)
p1.show()