from sympy import *

# (s^2) * (1+r) * (exp(-r))
# where
# 	r = sqrt(3) * (a-b)(a-b) / l

# (s^2) * (1+(sqrt(3) * (a-b)(a-b) / l)) * (exp(-(sqrt(3) * (a-b)(a-b) / l)))

#f = (s^2) * (1+(sqrt(3) * (a-b)(a-b) / l)) * (exp(-(sqrt(3) * (a-b)(a-b) / l)))

a, b, s, l = symbols('a b s l')

# diff( (s^2) * (1+(sqrt(3) * (a-b)*(a-b) / l)) * (exp(-(sqrt(3) * (a-b)*(a-b) / l))), a)

out = diff( ( (s^2) * (1+(sqrt(3) * (a-b)*(a-b) / l))) , a)
print(out)