import numpy as np
from scipy.optimize import minimize
import solver_constlump as solver  # Make sure this points to your solver file.

np.set_printoptions(suppress=True, precision=8)

# 1. Parameters
xi = 0.1
theta = 1.0
n = 5

# CHANGED: Lower G from 3.0 -> 1.0 for easier feasibility
G = 1.0

# CHANGED: Narrowed phi to keep productivities closer together
# (Original might have gone up to 2.5 or more)
phi = np.array([0.4, 0.6, 0.8, 1.0, 1.2])

# Keep lumpsum shares = 0.2 for every household
l_fixed = np.array([0.2]*n)

# 2. Objective Function (Social Welfare)
def swf_obj(x):
    # Decision variables: tau_w for each household and tau_z
    tau_w = x[0:n]
    tau_z = x[n]
    
    # Solve the economic model with these taxes
    utilities, agg_polluting, converged, c, d, ell, w, p_d, l_val, _ = solver.solve(tau_w, tau_z, G, n=n)
    if not converged:
        return 1e10  # Penalize non-convergence
    
    # Sum of utilities minus pollution cost
    welfare = np.sum(utilities) - xi * (agg_polluting**theta)
    return -welfare  # Negative for minimization

# 3. Mirrlees IC Constraints
def ic_constraints(x):
    tau_w = x[0:n]
    tau_z = x[n]
    
    utilities, agg_polluting, converged, c, d, ell, w, p_d, l_val, _ = solver.solve(tau_w, tau_z, G, n=n)
    if not converged:
        return -np.ones(n*(n-1)) * 1e6

    # Utility params
    alpha, beta, gamma = 0.7, 0.2, 0.2
    d0 = 0.1
    T = 1.0
    
    # Income for each type j
    I = np.zeros(n)
    for j in range(n):
        I[j] = (T - ell[j]) * (1.0 - tau_w[j]) * phi[j] * w + l_val

    # Build list of U_i - U_{i_j} for all i != j
    g_list = []
    for i in range(n):
        U_i = utilities[i]  # from solver
        for j in range(n):
            if i == j:
                continue
            
            c_j = c[j]
            d_j = d[j]
            
            denom = (1.0 - tau_w[j]) * phi[i] * w
            if denom <= 0:
                g_list.append(-1e6)
                continue
            
            ell_i_j = T - I[j] / denom
            if ell_i_j <= 0 or c_j <= 0 or d_j <= d0:
                # invalid => big negative penalty
                U_i_j = -1e6
            else:
                U_i_j = (alpha * np.log(c_j)
                         + beta * np.log(d_j - d0)
                         + gamma * np.log(ell_i_j))
            
            # i's true utility minus i's masquerade-as-j utility
            g_list.append(U_i - U_i_j)
    
    return np.array(g_list)

# 4. Solve the Planner Problem
initial_tau_w = [0.05]*n
initial_tau_z = [0.5]
initial_guess = np.array(initial_tau_w + initial_tau_z)

# CHANGED: Tighter bounds so solver doesn't try huge or negative taxes
bounds = [(0.0, 1.0)]*n + [(0.01, 5.0)]

constraints = [
    {'type': 'ineq', 'fun': ic_constraints}
]

res = minimize(swf_obj, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

if res.success:
    opt_tau_w = res.x[0:n]
    opt_tau_z = res.x[n]
    max_welfare = -res.fun
    print("Social Welfare Optimization Successful!")
    print("Optimal tau_w:", opt_tau_w)
    print("Optimal tau_z:", opt_tau_z)
    print("Fixed lump-sum shares:", l_fixed)
    print("Maximized Social Welfare:", max_welfare)
    print("Solver message:", res.message)
else:
    print("Optimization Failed!")
    print("Solver message:", res.message)