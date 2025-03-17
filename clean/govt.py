import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import solver

np.set_printoptions(suppress=True, precision=8)

# 1. Parameters
xi = 10.0
theta = 1.0
n = 5
G = 1.0
phi = np.array([0.03*5, 0.0825*5, 0.141*5, 0.229*5, 0.511*5])

# 2. Objective Function (Social Welfare)
def swf_obj(x):
    # Extract decision variables:
    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]
    
    # Solve the economic model:
    utilities, agg_polluting, converged, c, d, ell, w, p_d = solver.solve(tau_w, tau_z, l, G, n=n)
    if not converged:
        return 1e10  # Penalize non-convergence
    
    # Compute welfare:
    welfare = np.sum(utilities) - xi * (agg_polluting**theta)
    return -welfare  # Negative because we minimize

# 3. Mirrlees IC Constraints
def ic_constraints(x):
    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]
    
    utilities, agg_polluting, converged, c, d, ell, w, p_d = solver.solve(tau_w, tau_z, l, G, n=n)
    if not converged:
        return -np.ones(n*(n-1)) * 1e6  # Large negative penalty if not converged

    # Parameters for utility
    alpha, beta, gamma = 0.7, 0.2, 0.2
    d0 = 0.1
    T = 1.0

    # Compute income measure I for each type:
    I = np.zeros(n)
    for j in range(n):
        I[j] = (T - ell[j])*(1.0 - tau_w[j])*phi[j]*w + l[j]*tau_z*agg_polluting

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
            
            ell_i_j = T - I[j] / denom
            if ell_i_j <= 0:
                U_i_j = -1e6
            else:
                if c_j <= 0 or d_j <= d0:
                    U_i_j = -1e6
                else:
                    U_i_j = (alpha * np.log(c_j) +
                             beta * np.log(d_j - d0) +
                             gamma * np.log(ell_i_j))
            
            # Constraint: U_i must be at least U_i_j
            g_list.append(U_i - U_i_j)
    
    return np.array(g_list)

# 4. Government Spending Constraint
def gov_constraint(x):
    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]
    
    _, agg_polluting, converged, c, d, ell, w, p_d = solver.solve(tau_w, tau_z, l, G, n=n)
    return np.sum(tau_w * phi * w) + tau_z * agg_polluting * (1 - np.sum(l)) - G

# 5. Solve the Planner Problem
# Initial guesses: note lump-sum shares sum to 1.
initial_tau_w = [0.05] * n
initial_tau_z = [3.0]
initial_l = [1.0 / n] * n
initial_guess = np.array(initial_tau_w + initial_tau_z + initial_l)

# Define bounds for each variable:
bounds = [(-5.0, 5.0)] * n + [(0.1, 100.0)] + [(-5.0, 5.0)] * n

constraints = [
    #{'type': 'eq', 'fun': lambda x: np.sum(x[n+1:2*n+1]) - 1.0},  # Lump-sum shares constraint
    {'type': 'eq', 'fun': gov_constraint},                        # Gov spending constraint
    {'type': 'ineq', 'fun': ic_constraints}                         # IC constraints
]

res = minimize(swf_obj, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

# Output results
if res.success:
    opt_tau_w = res.x[0:n]
    opt_tau_z = res.x[n]
    opt_l = res.x[n+1:2*n+1]
    max_welfare = -res.fun
    print("Social Welfare Optimization Successful!")
    print("Optimal tau_w:", opt_tau_w)
    print("Optimal tau_z:", opt_tau_z)
    print("Optimal lump-sum shares:", opt_l)
    print("Maximized Social Welfare:", max_welfare)
    print("Solver message:", res.message)
else:
    print("Optimization Failed!")
    print("Solver message:", res.message)