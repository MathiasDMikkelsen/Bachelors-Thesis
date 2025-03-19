import numpy as np
from scipy.optimize import root

# Global model parameters (from your LaTeX)
alpha     = 0.7
beta      = 0.2
gamma     = 0.2
r         = -1.0
T_val     = 100.0      # Time endowment per household
L         = 0.0      # Lump-sum transfers
D0        = 1.5
epsilon_C = 0.995    # Example value for Sector C
epsilon_D = 0.92     # Example value for Sector D
p_C       = 1.0      # Numeraire
tau_z     = 1.0 
G         = 6.0


phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
n = len(phi)

def system_eqns(x):
    T_C, T_D, Z_C, Z_D, w, p_D, L = x

    F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
    F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)
    
    D_agents = (beta/(p_D*(alpha+beta+gamma)))*(phi*w*(1-tau_w)*T_val + L - p_D*D0) + D0
    l_agents = (gamma/((alpha+beta+gamma)*phi*w))*(phi*w*(1-tau_w)*T_val + L - p_D*D0)

    agg_labor = np.sum(phi*(T_val - l_agents))  
    agg_D = np.sum(D_agents)                     

    eq1 = T_C + T_D - agg_labor

    eq2 = (agg_D+G/p_D) - F_D 
    
    eq3 = w - epsilon_C * (T_C**(r-1)) * (F_C**(1-r)) 

    eq4 = tau_z - (1 - epsilon_C) * (Z_C**(r-1)) * (F_C**(1-r))
    
    eq5 = w - epsilon_D * (T_D**(r-1)) * (F_D**(1-r)) * p_D
    eq6 = tau_z - (1 - epsilon_D) * (Z_D**(r-1)) * (F_D**(1-r)) * p_D
    
    eq7 = (L-(np.sum(tau_w*w*phi*l_agents)+tau_z*(Z_C+Z_D)-G))/n

    return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7])

x0 = np.array([0.3, 0.4, 0.6, 0.4, 0.5, 1.5, 0.1])

sol = root(system_eqns, x0, method='lm')

print("Solution status:", sol.status)
print("Solution message:", sol.message)
print("Solution vector [T_C, T_D, Z_C, Z_D, w, p_D, L]:")
print(sol.x)

print("\nProduction Summary:")
T_C, T_D, Z_C, Z_D, w, p_D, L = sol.x
print(f"Sector C: T_prod = {T_C:.4f}, Z_C = {Z_C:.4f}")
print(f"Sector D: T_prod = {T_D:.4f}, Z_D = {Z_D:.4f}")
print(f"Common wage, w = {w:.4f}, p_D = {p_D:.4f}")

F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)
print(f"Sector C output, F_C = {F_C:.4f}")
print(f"Sector D output, F_D = {F_D:.4f}")

# --- Compute Household-Level Demands and Aggregates ---
C_agents = (alpha/(p_C*(alpha+beta+gamma)))*(phi*w*T_val + L - p_D*D0)
D_agents = (beta/(p_D*(alpha+beta+gamma)))*(phi*w*T_val + L - p_D*D0) + D0
l_agents = (gamma/((alpha+beta+gamma)*phi*w))*(phi*w*T_val + L - p_D*D0)

print("\nHousehold Demands and Leisure:")
for i in range(n):
    print(f"Household {i+1}: C = {C_agents[i]:.4f}, D = {D_agents[i]:.4f}, l = {l_agents[i]:.4f}")

agg_C = np.sum(C_agents)
agg_D = np.sum(D_agents)
agg_labor = np.sum(phi*(T_val - l_agents))
print("\nAggregated Quantities:")
print(f"Aggregate C = {agg_C:.4f}")
print(f"Aggregate D = {agg_D:.4f}")
print(f"Labor supply = {agg_labor:.4f}")

print("\nLump sum:")
print(f"L = {L:.4f}")

# --- Print Residuals ---
residuals = system_eqns(sol.x)
labels = [
    "Labor market equilibrium residual",
    "Good D market clearing residual",
    "Sector C wage FOC residual",
    "Sector C margin FOC residual",
    "Sector D wage FOC residual",
    "Sector D margin FOC residual"
]

print("\nResiduals (should be close to zero):")
for label, res in zip(labels, residuals):
    print(f"{label}: {res:.10f}")
