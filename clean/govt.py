import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import solver

# 1. params
xi = 3.0
theta = 1.0
n = 5   
G = 1.0
phi = np.array([0.1*5, 0.1*5, 0.2*5, 0.3*5, 0.511*5])
# end params

# 2. objective
def swf_obj(x):

    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]
    
    # Ensure l sums to 1.0
    if not np.isclose(np.sum(l), 1.0):
        return 1e10
    
    # Solve the equilibrium
    utilities, agg_polluting, converged, c, d, ell, w = solver.solve(tau_w, tau_z, l, G, n=n)
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
    
    utilities, agg_polluting, converged, c, d, ell, w = solver.solve(tau_w, tau_z, l, G, n=n)
    if not converged:
        return -np.ones(n*(n-1)) * 1e6
    
    alpha, beta, gamma = 0.7, 0.2, 0.2
    d0 = 0.1
    T = 1.0  # total time endowment
    
    I = np.zeros(n)
    for j in range(n):
        I[j] = (T - ell[j])*(1.0 - tau_w[j])*phi[j]*w
    
    # Build a list for g_{ij} = U_i - U_i^j
    g_list = []
    
    for i in range(n):
        U_i = utilities[i]  # true utility of i
        
        for j in range(n):
            if i == j:
                continue
            
            c_j = c[j]
            d_j = d[j]
            
            denom = (1.0 - tau_w[j]) * phi[i] * w
            if denom <= 0:
                g_list.append(-1e6)
                continue
            
            ell_i_j = T - I[j]/denom
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
    
    return np.array(g_list)  # shape = (n*(n-1),)

def gov_constraint(x):
    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]
    
    _, agg_polluting, converged, c, d, ell, w = solver.solve(tau_w, tau_z, l, G, n=n)
    return np.sum(tau_w * phi * w) + tau_z * agg_polluting *(1- np.sum(l)) - G

###############################################################################
# 5) Setup and Solve the Government's Problem
###############################################################################
# Initial guess: tau_w (n), tau_z, l_vector (n)
initial_guess = np.array([0.05]*n + [3.0] + [0.01]*n)

# Bounds for each variable
bounds = [(0.0, 1.0)]*n + [(0.1, 100.0)] + [(0.0, 1.0)]*n

# Constraints:
# 1) sum of l_i = 1
# 2) sum(tau_w[i]*phi[i]*w[i]) = G
# 3) Mirrlees-type IC constraints: U_i - U_i^j >= 0
constraints = [
    {'type': 'eq', 'fun': lambda x: np.sum(x[n+1:2*n+1]) - 1.0},
    {'type': 'eq', 'fun': gov_constraint}, # we need to add revenues from aggregare pollution here and then substract them again
    {'type': 'ineq', 'fun': ic_constraints}
]

# Solve using SLSQP
res = minimize(swf_obj, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

opt_tau_w = res.x[0:n]
opt_tau_z = res.x[n]
opt_l = res.x[n+1:2*n+1]
max_welfare = -res.fun

print("\n=== Social Welfare Optimization with Mirrlees IC Constraint ===")
print("Optimal tau_w:", opt_tau_w)
print("Optimal tau_z:", opt_tau_z)
print("Optimal l_vector:", opt_l)
print("Maximized Social Welfare:", max_welfare)
print("Convergence:", res.success)
print("Message:", res.message)