"""
File for executing random tests of casadi etc.
"""
import casadi as ca

q = ca.MX.sym('q', 4) 
qg = ca.MX.sym('qg', 4) 

dot = qg.T @ q
dot_scalar = dot[0, 0]
j_geo = 1 - ca.fabs(dot_scalar)

grad_j_geo = ca.gradient(j_geo, q)

hess_j_geo = ca.hessian(j_geo, q)

print("Gradient:")
print(grad_j_geo)

print("Hessian:")
print(hess_j_geo)
# hessian returns 0, we'd need to hardcode it