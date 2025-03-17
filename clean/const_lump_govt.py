import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import solver_constlump as solver

np.set_printoptions(suppress=True, precision=8)

# 1. Parameters
xi = 10.0
theta = 1.0
n = 5
G = 1.0
phi = np.array([0.03*5, 0.0825*5, 0.141*5, 0.229*5, 0.5175*5])

# 2. Objective Function (Social Welfare)
def swf_obj(x):
    # Extract decision variables:
    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]
    
    # Solve the economic model:
    utilities, agg_polluting, converged, c, d, ell, w, p_d, l_val, _ = solver.solve(tau_w, tau_z, G, n=n)
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
    
    utilities, agg_polluting, converged, c, d, ell, w, p_d, l_val, _ = solver.solve(tau_w, tau_z, G, n=n)
    if not converged:
        return -np.ones(n*(n-1)) * 1e6  # Large negative penalty if not converged

    # Parameters for utility
    alpha, beta, gamma = 0.7, 0.2, 0.2
    d0 = 0.1
    T = 1.0
    
    # Compute income measure I for each type:
    I = np.zeros(n)
    for j in range(n):
        I[j] = (T - ell[j])*(1.0 - tau_w[j])*phi[j]*w + l_val

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


# 5. Solve the Planner Problem
# Since lump sum shares should not sum to 1, we choose an initial guess that does not sum to 1.
initial_tau_w = [0.05] * n
initial_tau_z = [3.0]
initial_l = [0.2] * n   # Sum is 0, not 1
initial_guess = np.array(initial_tau_w + initial_tau_z + initial_l)

# Define bounds for each variable:
bounds = [(-2.0, 2.0)] * n + [(0.1, 100.0)] + [(-2.0, 2.0)] * n

# For now we only enforce the government spending constraint; you can enable the IC constraints as needed.
constraints = [
    #{'type': 'eq', 'fun': gov_constraint},
    {'type': 'ineq', 'fun': ic_constraints}  # Uncomment if you want to enforce IC constraints.
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
    