import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import solver

np.set_printoptions(suppress=True, precision=8)

# 1. params
xi = 1.0
theta = 1.0
n = 5   
G = 2.0
phi = np.array([0.03*5, 0.0825*5, 0.141*5, 0.229*5, 0.511*5])
# end params

# 2. objective
def swf_obj(x):

    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]
    
    if not np.isclose(np.sum(l), 1.0):
        return 1e10
    
    utilities, agg_polluting, converged, c, d, ell, w, p_d = solver.solve(tau_w, tau_z, l, G, n=n)
    if not converged:
        return 1e10
    
    welfare = np.sum(utilities) - xi*(agg_polluting**theta)
    return -welfare 
# end defining objective function

# 3. mirrlees ic constraints
def ic_constraints(x):
    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]
    
    utilities, agg_polluting, converged, c, d, ell, w, p_d = solver.solve(tau_w, tau_z, l, G, n=n)
    if not converged:
        return -np.ones(n*(n-1)) * 1e6
    
    alpha, beta, gamma = 0.7, 0.2, 0.2
    d0 = 0.1
    T = 1.0 
    
    #I = np.zeros(n)
    #for j in range(n):
     #   I[j] = (T - ell[j])*(1.0 - tau_w[j])*phi[j]*w+l[j]*tau_z*agg_polluting
    
    g_list = []
    
    for i in range(n):
        U_i = utilities[i]  
        
        for j in range(n):
            if i == j:
                continue
            
            c_j = c[j]
            d_j = d[j]
            
            denom = (1.0 - tau_w[j]) * phi[i] * w
            if denom <= 0:
                g_list.append(-1e6)
                continue
            
            ell_i_j = T- (c[j]+p_d*d[j])/denom
            if ell_i_j <= 0:
                U_i_j = -1e6
            else:
                if c_j <= 0 or (d_j <= d0):
                    U_i_j = -1e6
                else:
                    U_i_j = (alpha*np.log(c_j) +
                             beta*np.log(d_j - d0) +
                             gamma*np.log(ell_i_j))
            
            g_list.append(U_i - U_i_j)
    
    return np.array(g_list)
# end ic constraint

# 4. government spending constraint
def gov_constraint(x):
    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]
    
    _, agg_polluting, converged, c, d, ell, w, p_d = solver.solve(tau_w, tau_z, l, G, n=n)
    return np.sum(tau_w * phi * w) + tau_z * agg_polluting *(1- np.sum(l)) - G
# end government spending constraint

# 5. solve planner problem
initial_guess = np.array([0.05]*n + [0.1] + [0.01]*n)

bounds = [(0.0, 1.0)]*n + [(0.1, 100.0)] + [(0.0, 1.0)]*n

constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x[n+1:2*n+1]) - 1.0},
    {'type': 'eq', 'fun': gov_constraint},
    {'type': 'ineq', 'fun': ic_constraints}
]

res = minimize(swf_obj, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

opt_tau_w = res.x[0:n]
opt_tau_z = res.x[n]
opt_l = res.x[n+1:2*n+1]
max_welfare = -res.fun

print("sw optimization")
print("tau_w:", opt_tau_w)
print("tau_z:", opt_tau_z)
print("lump-sum:", opt_l)
print("max sw:", max_welfare)
print("convergence:", res.success)
print("message:", res.message)