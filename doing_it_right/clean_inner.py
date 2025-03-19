import numpy as np
from scipy.optimize import root

# Global model parameters (from your LaTeX)
alpha     = 0.7
beta      = 0.2
gamma     = 0.2
r         = -1.0
T_val     = 1000.0      # Time endowment per household
L_global  = 0.0         # Lump-sum transfers (but L is solved for)
D0        = 1.5
epsilon_C = 0.995       # Example value for Sector C
epsilon_D = 0.92        # Example value for Sector D
p_C       = 1.0         # Numeraire
G         = 6.0

# Household weights (for 5 households, these sum to 1)
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n = len(phi)    

# The equilibrium system. Note that we now add tau_z and tau_w as extra arguments.
# The unknown equilibrium variables are:
# x = [T_C, T_D, Z_C, Z_D, w, p_D, L]
def system_eqns(x, tau_z, tau_w):
    
    # Global model parameters (from your LaTeX)
    alpha     = 0.7
    beta      = 0.2
    gamma     = 0.2
    r         = -1.0
    T_val     = 1000.0      # Time endowment per household
    L_global  = 0.0         # Lump-sum transfers (but L is solved for)
    D0        = 1.5
    epsilon_C = 0.995       # Example value for Sector C
    epsilon_D = 0.92        # Example value for Sector D
    p_C       = 1.0         # Numeraire
    G         = 6.0

    # Household weights (for 5 households, these sum to 1)
    phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
    n = len(phi)    
    
    T_C, T_D, Z_C, Z_D, w, p_D, L = x

    F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
    F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)
    
    # Household-level demands, per your LaTeX:
    # Note: The effective wage for each household is w*(1-tau_w) and tau_w is applied elementwise.
    C_agents = (alpha/(p_C*(alpha+beta+gamma)))*(phi * w * (1 - tau_w) * T_val + L - p_D*D0)
    D_agents = (beta/(p_D*(alpha+beta+gamma)))*(phi * w * (1 - tau_w) * T_val + L - p_D*D0) + D0
    l_agents = (gamma/((alpha+beta+gamma)*phi*w*(1 - tau_w)))*(phi * w * (1 - tau_w) * T_val + L - p_D*D0)
    
    # Aggregated outcomes:
    agg_labor = np.sum(phi * (T_val - l_agents))
    agg_D = np.sum(D_agents)
    
    # Equilibrium conditions:
    eq1 = T_C + T_D - agg_labor                      # Labor market equilibrium
    eq2 = agg_D - F_D                              # Market clearing for good D

    # Production FOCs for Sector C:
    eq3 = w - epsilon_C * (T_C**(r-1)) * (F_C**(1-r))      # Wage FOC (p_C = 1)
    eq4 = tau_z - (1 - epsilon_C) * (Z_C**(r-1)) * (F_C**(1-r))  # Margin FOC

    # Production FOCs for Sector D:
    eq5 = w - epsilon_D * (T_D**(r-1)) * (F_D**(1-r)) * p_D  # Wage FOC
    eq6 = tau_z - (1 - epsilon_D) * (Z_D**(r-1)) * (F_D**(1-r)) * p_D  # Margin FOC

    # Lump-sum equilibrium condition:
    # Here, we assume the market-clearing condition:
    # n*L = sum(tau_w * w * phi * l_agents) + tau_z*(Z_C+Z_D) - G
    eq7 = n * L - (np.sum(tau_w * w * phi * l_agents) + tau_z * (Z_C + Z_D) - G)

    return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7])

# Now we wrap everything in a function that takes tau_z and tau_w as input.
def solve_and_return(tau_z, tau_w):
    # tau_w should be a numpy array of length 5; if a scalar is passed, broadcast it.
    tau_w = np.asarray(tau_w)
    if tau_w.size == 1:
        tau_w = np.full(n, tau_w)
    
    # Initial guess for x = [T_C, T_D, Z_C, Z_D, w, p_D, L]
    x0 = np.array([0.3, 0.4, 0.6, 0.4, 0.5, 1.5, 0.1])
    
    sol = root(system_eqns, x0, args=(tau_z, tau_w), method='lm')
    if not sol.success:
        print("Equilibrium did not converge for tau_z =", tau_z, "and tau_w =", tau_w)
        return None
    
    # Print equilibrium solution:
    print("Solution status:", sol.status)
    print("Solution message:", sol.message)
    print("Solution vector [T_C, T_D, Z_C, Z_D, w, p_D, L]:")
    print(sol.x)
    
    # Production summary:
    T_C, T_D, Z_C, Z_D, w, p_D, L = sol.x
    F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
    F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)
    print("\nProduction Summary:")
    print(f"Sector C: T_prod = {T_C:.4f}, Z_C = {Z_C:.4f}, F_C = {F_C:.4f}")
    print(f"Sector D: T_prod = {T_D:.4f}, Z_D = {Z_D:.4f}, F_D = {F_D:.4f}")
    print(f"Common wage, w = {w:.4f}, p_D = {p_D:.4f}, L = {L:.4f}")
    
    # Compute equilibrium residuals:
    resids = system_eqns(sol.x, tau_z, tau_w)
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
        print(f"{label}: {r_val:.15f}")
    
    # Effective wage for households:
    w_eff = w * (1 - tau_w)  # elementwise
    
    # Compute household-level demands:
    C_agents = (alpha/(p_C*(alpha+beta+gamma)))*(phi * w_eff * T_val + L - p_D*D0)
    D_agents = (beta/(p_D*(alpha+beta+gamma)))*(phi * w_eff * T_val + L - p_D*D0) + D0
    l_agents = (gamma/((alpha+beta+gamma)*phi*w_eff))*(phi * w_eff * T_val + L - p_D*D0)
    
    print("\nHousehold Equilibrium Outcomes:")
    for i in range(n):
        print(f"Household {i+1}: C = {C_agents[i]:.4f}, D = {D_agents[i]:.4f}, l = {l_agents[i]:.4f}")
    
    # Compute sum of household utility:
    # U_i = C_i^alpha + (D_i - D0)^beta + l_i^gamma
    U_agents = C_agents**alpha + (D_agents - D0)**beta + l_agents**gamma
    sum_utility = np.sum(U_agents)
    print("\nSum of household utilities:", sum_utility)
    
    # Compute sum of complementary inputs:
    sum_Z = Z_C + Z_D
    print("Sum of Z_C and Z_D:", sum_Z)
    
    return sum_utility, sum_Z

# Example call:
# Use tau_z = 0.02 and tau_w = 0.0 (applied to all households)
result = solve_and_return(tau_z=0.02, tau_w=0.0)
print("\nReturned values (sum utility, sum of Z's):", result)