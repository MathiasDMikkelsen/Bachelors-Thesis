import numpy as np
from scipy.optimize import root

# Global model parameters (from your LaTeX)
alpha     = 0.7
beta      = 0.2
gamma     = 0.2
r         = -1.0
T_val     = 100.0      # Time endowment per household
L_global  = 0.0         # Lump-sum transfers (but L is solved for)
D0        = 1.5
epsilon_C = 0.995       # Example value for Sector C
epsilon_D = 0.92        # Example value for Sector D
p_C       = 1.0         # Numeraire
tau_z_default = 1.0     # Default tau_z (will be replaced by input)
G         = 6.0

# Household weights (for 5 households, these sum to 1)
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n = len(phi)

def system_eqns(x, tau_z_input, tau_w_input):
    T_C, T_D, Z_C, Z_D, w, p_D, L = x
    F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
    F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)
    # Household-level demands:
    # Each household i:
    #   C_i = φ_i * [α/(p_C(α+β+γ))]{ w*(1-tau_w_i)*T_val + L - p_D*D0 }
    #   D_i = φ_i * [β/(p_D(α+β+γ))]{ w*(1-tau_w_i)*T_val + L - p_D*D0 } + D0
    #   l_i = φ_i * [γ/((α+β+γ)*w*(1-tau_w_i))]{ w*(1-tau_w_i)*T_val + L - p_D*D0 }
    C_agents = phi * (alpha/(p_C*(alpha+beta+gamma))) * (w*(1-tau_w_input)*T_val + L - p_D*D0)
    D_agents = phi * (beta/(p_D*(alpha+beta+gamma))) * (w*(1-tau_w_input)*T_val + L - p_D*D0) + D0
    l_agents = phi * (gamma/((alpha+beta+gamma)*w*(1-tau_w_input))) * (w*(1-tau_w_input)*T_val + L - p_D*D0)
    
    agg_labor = np.sum(T_val - l_agents)
    agg_D = np.sum(D_agents)
    
    eq1 = T_C + T_D - agg_labor
    eq2 = agg_D - ((epsilon_D*(T_D**r) + (1-epsilon_D)*(Z_D**r))**(1/r))
    eq3 = w - epsilon_C * (T_C**(r-1)) * ((epsilon_C*(T_C**r) + (1-epsilon_C)*(Z_C**r))**((1-r)/r))
    eq4 = tau_z_input - (1 - epsilon_C) * (Z_C**(r-1)) * ((epsilon_C*(T_C**r) + (1-epsilon_C)*(Z_C**r))**((1-r)/r))
    eq5 = w - epsilon_D * (T_D**(r-1)) * ((epsilon_D*(T_D**r) + (1-epsilon_D)*(Z_D**r))**((1-r)/r)) * p_D
    eq6 = tau_z_input - (1 - epsilon_D) * (Z_D**(r-1)) * ((epsilon_D*(T_D**r) + (1-epsilon_D)*(Z_D**r))**((1-r)/r)) * p_D
    eq7 = n*L - (np.sum(tau_w_input * w * phi * l_agents) + tau_z_input*(Z_C+Z_D) - G)
    return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7])

def solve_and_return(tau_z_input, tau_w_input):

    # Ensure tau_w_input is an array of length n:
    tau_w_input = np.asarray(tau_w_input)
    if tau_w_input.size == 1:
        tau_w_input = np.full(n, tau_w_input)
    
    x0 = np.array([0.3, 0.4, 0.6, 0.4, 0.5, 1.5, 0.1])
    sol = root(system_eqns, x0, args=(tau_z_input, tau_w_input), method='lm')
    if not sol.success:
        print("Equilibrium did not converge for tau_z =", tau_z_input, "and tau_w =", tau_w_input)
        return None

    print("Solution status:", sol.status)
    print("Solution message:", sol.message)
    print("Solution vector [T_C, T_D, Z_C, Z_D, w, p_D, L]:")
    print(sol.x)
    
    T_C, T_D, Z_C, Z_D, w, p_D, L = sol.x
    F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
    F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)
    print("\nProduction Summary:")
    print(f"Sector C: T_prod = {T_C:.4f}, Z_C = {Z_C:.4f}, F_C = {F_C:.4f}")
    print(f"Sector D: T_prod = {T_D:.4f}, Z_D = {Z_D:.4f}, F_D = {F_D:.4f}")
    print(f"Common wage, w = {w:.4f}, p_D = {p_D:.4f}, L = {L:.4f}")
    
    resids = system_eqns(sol.x, tau_z_input, tau_w_input)
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
    
    # Compute household outcomes:
    # For each household, effective wage = w*(1-tau_w_i)
    w_eff = w * (1 - tau_w_input)
    C_agents = phi * (alpha/(p_C*(alpha+beta+gamma))) * (w*(1-tau_w_input)*T_val + L - p_D*D0)
    D_agents = phi * (beta/(p_D*(alpha+beta+gamma))) * (w*(1-tau_w_input)*T_val + L - p_D*D0) + D0
    l_agents = phi * (gamma/((alpha+beta+gamma)*w*(1-tau_w_input))) * (w*(1-tau_w_input)*T_val + L - p_D*D0)
    
    print("\nHousehold Demands and Leisure:")
    for i in range(n):
        print(f"Household {i+1}: C = {C_agents[i]:.4f}, D = {D_agents[i]:.4f}, l = {l_agents[i]:.4f}")
    
    U_agents = C_agents**alpha + (D_agents-D0)**beta + l_agents**gamma
    sum_util = np.sum(U_agents)
    print("\nSum of household utilities:", sum_util)
    
    sum_Z = Z_C + Z_D
    print("Sum of Z_C and Z_D:", sum_Z)
    
    return sum_util, sum_Z


result = solve_and_return(0.3, tau_w_input=np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
print("\nReturned values (sum utility, sum of Z's):", result)