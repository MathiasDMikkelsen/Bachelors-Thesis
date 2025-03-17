import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# 1) Model Parameters
xi = 3.0
theta = 1.0
n = 5   # number of households
G = 1.0 # sum_{i} tau_w[i]*phi[i]*w[i] must equal G
phi = np.ones(n)  # participation factors (adjust as needed)
w   = np.ones(n)  # wages (adjust as needed)

###############################################################################
# 2) Placeholder for your extended jeppes_method
###############################################################################
def main_solve_print_extended(tau_w, tau_z, l, n=5):
    """
    This function should:
      1) Solve the equilibrium (just like main_solve_print).
      2) Return:
          - utilities[i] = U_i,
          - c[i], d[i], ell[i] for each household,
          - a boolean flag 'converged' indicating solver success.
    
    For demonstration, we mock up some arrays. You must replace this
    with your actual model solution. 
    """
    # In a real version, you'd solve your 21-equation system here.
    # We'll produce dummy values for demonstration only:
    converged = True  # or False if your root finder fails
    
    # Suppose each household ends up with these dummy results:
    c = np.array([1.0, 1.2, 1.4, 1.6, 1.8])  # consumption of the clean good
    d = np.array([0.8, 0.9, 1.0, 1.1, 1.2])  # polluting good
    ell = np.array([0.5, 0.4, 0.3, 0.35, 0.45])  # leisure
    
    # Utility function: U_i = alpha ln(c_i) + beta ln(d_i - d0) + gamma ln(ell_i)
    # (Just an example; adapt to your real code.)
    alpha, beta, gamma, d0 = 0.7, 0.2, 0.2, 0.01
    utilities = np.zeros(n)
    for i in range(n):
        if c[i] <= 0 or d[i] <= d0 or ell[i] <= 0:
            utilities[i] = -1e6  # penalty for invalid
        else:
            utilities[i] = (alpha*np.log(c[i]) +
                            beta*np.log(d[i] - d0) +
                            gamma*np.log(ell[i]))
    
    # Let's say "aggregate_polluting" is sum of polluting inputs in the economy
    aggregate_polluting = 2.0  # dummy value
    
    return utilities, aggregate_polluting, converged, c, d, ell

###############################################################################
# 3) Social Welfare Objective
###############################################################################
def swf_obj(x):
    """
    Decision variables:
      x[0:n]      = tau_w (tax rates for each household)
      x[n]        = tau_z (pollution tax)
      x[n+1:2*n+1]= l (lump-sum transfers)
    
    Returns negative of social welfare = - [ sum of utilities - xi*(pollution^theta) ].
    """
    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]
    
    # Ensure l sums to 1.0
    if not np.isclose(np.sum(l), 1.0):
        return 1e10
    
    # Solve the equilibrium
    utilities, agg_polluting, converged, c, d, ell = main_solve_print_extended(tau_w, tau_z, l, n=n)
    if not converged:
        return 1e10
    
    welfare = np.sum(utilities) - xi*(agg_polluting**theta)
    return -welfare  # negative for minimization

###############################################################################
# 4) Mirrlees-Type IC Constraints
###############################################################################
def ic_constraints(x):
    tau_w = x[0:n]
    tau_z = x[n]
    l = x[n+1:2*n+1]
    
    # Solve for the equilibrium details
    utilities, agg_polluting, converged, c, d, ell = main_solve_print_extended(tau_w, tau_z, l, n=n)
    if not converged:
        # If not converged, force a large negative to break feasibility
        return -np.ones(n*(n-1)) * 1e6
    
    # Basic utility parameters (must match your model):
    alpha, beta, gamma = 0.7, 0.2, 0.2
    d0 = 0.01
    T = 1.0  # total time endowment
    
    # Compute net labor income for each j:
    #   I_j = (T - ell_j)*(1 - tau_w[j])*phi[j]*w[j]
    I = np.zeros(n)
    for j in range(n):
        I[j] = (T - ell[j])*(1.0 - tau_w[j])*phi[j]*w[j]
    
    # Build a list for g_{ij} = U_i - U_i^j
    g_list = []
    
    for i in range(n):
        U_i = utilities[i]  # true utility of i
        
        for j in range(n):
            if i == j:
                continue
            
            # c_j, d_j from the equilibrium
            c_j = c[j]
            d_j = d[j]
            
            # If i pretends to be j, i must supply labor so that
            # (T - ell_i^j)*(1 - tau_w[j])*phi[i]*w[i] = I_j
            denom = (1.0 - tau_w[j]) * phi[i] * w[i]
            if denom <= 0:
                # If denom is zero or negative, treat as invalid => big negative
                g_list.append(-1e6)
                continue
            
            ell_i_j = T - I[j]/denom
            if ell_i_j <= 0:
                # invalid leisure => utility is negative infinity
                U_i_j = -1e6
            else:
                # Utility if i uses c_j, d_j, and ell_i_j
                if c_j <= 0 or (d_j <= d0):
                    U_i_j = -1e6
                else:
                    U_i_j = (alpha*np.log(c_j) +
                             beta*np.log(d_j - d0) +
                             gamma*np.log(ell_i_j))
            
            g_list.append(U_i - U_i_j)
    
    return np.array(g_list)  # shape = (n*(n-1),)

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
    {'type': 'eq', 'fun': lambda x: np.sum(x[0:n]*phi*w) - G}, # we need to add revenues from aggregare pollution here and then substract them again
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