import numpy as np
from scipy.optimize import root, minimize
import clean_solver as solver

def solve_and_return(tau_z, tau_w):
    # Ensure tau_w is an array of length n:
    tau_w = np.asarray(tau_w)
    if tau_w.size == 1:
        tau_w = np.full(n, tau_w)
    
    # Initial guess for x = [T_C, T_D, Z_C, Z_D, w, p_D, L]
    x0 = np.array([0.3, 0.4, 0.6, 0.4, 0.5, 1.5, 0.1])
    sol = root(equilibrium_system, x0, args=(tau_w, tau_z), method='lm')
    if not sol.success:
        print("Equilibrium did not converge for tau_z =", tau_z, "and tau_w =", tau_w)
        return None
    # Print equilibrium solution:
    print("Equilibrium solution (x = [T_C, T_D, Z_C, Z_D, w, p_D, L]):")
    print(sol.x)
    T_C, T_D, Z_C, Z_D, w, p_D, L = sol.x
    F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
    F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)
    print("\nProduction Summary:")
    print(f"Sector C: T_prod = {T_C:.4f}, Z_C = {Z_C:.4f}, F_C = {F_C:.4f}")
    print(f"Sector D: T_prod = {T_D:.4f}, Z_D = {Z_D:.4f}, F_D = {F_D:.4f}")
    print(f"Common wage, w = {w:.4f}, p_D = {p_D:.4f}, L = {L:.4f}")
    
    # Print equilibrium residuals:
    resids = equilibrium_system(sol.x, tau_w, tau_z)
    labels = [
        "Labor market equilibrium residual",
        "Good D market clearing residual",
        "Sector C wage FOC residual",
        "Sector C margin FOC residual",
        "Sector D wage FOC residual",
        "Sector D margin FOC residual",
        "Lump-sum equilibrium residual"
    ]
    print("\nEquilibrium Residuals:")
    for label, r_val in zip(labels, resids):
        print(f"{label}: {r_val:.10f}")
    
    # Effective wage for households:
    w_eff = w * (1 - tau_w)  # elementwise

    # Household-level outcomes:
    C_agents = (alpha/(p_C*(alpha+beta+gamma)))*(phi * w_eff * T_val + L - p_D*D0)
    D_agents = (beta/(p_D*(alpha+beta+gamma)))*(phi * w_eff * T_val + L - p_D*D0) + D0
    l_agents = (gamma/((alpha+beta+gamma)*phi*w_eff))*(phi * w_eff * T_val + L - p_D*D0)
    
    print("\nHousehold Equilibrium Outcomes:")
    for i in range(n):
        print(f"Household {i+1}: C = {C_agents[i]:.4f}, D = {D_agents[i]:.4f}, l = {l_agents[i]:.4f}")
    
    # Compute sum of household utilities using U = C^alpha + (D-D0)^beta + l^gamma
    U_agents = C_agents**alpha + (D_agents-D0)**beta + l_agents**gamma
    sum_util = np.sum(U_agents)
    print("\nSum of household utilities:", sum_util)
    
    # Compute sum of complementary inputs used: Z_C + Z_D
    sum_Z = Z_C + Z_D
    print("Sum of Z_C and Z_D:", sum_Z)
    
    return sum_util, sum_Z

# ------------------------------------------------------------------
# Now, put everything inside a function that takes tau_z and tau_w as input,
# prints the solution, and returns [sum of utilities, sum of Z_C and Z_D].
def optimize_policy(tau_z, tau_w):
    # tau_w can be a scalar or array; here we expect an array of length 5.
    result = solve_and_return(tau_z, tau_w)
    return result

# Example usage:
# Suppose we want to evaluate the model at tau_z = 0.02 and for tau_w = 0.0 for each household.
result = optimize_policy(0.02, 0.0)
print("\nReturned values (sum utility, sum of Z's):", result)