import numpy as np
from scipy.optimize import root

def system_eqns(x):
    # Unpack unknowns:
    # x = [T_d, Z_d, T_c, Z_c, w, p_D]
    T_d = x[0]    # Sector D production time
    Z_d = x[1]    # Sector D complementary input
    T_c = x[2]    # Sector C production time
    Z_c = x[3]    # Sector C complementary input
    w   = x[4]    # Common wage
    p_D = x[5]    # Price in sector D (p_C = 1)
    
    # --- Parameters ---
    alpha     = 0.7
    beta      = 0.2
    gamma     = 0.2
    r         = -1.0
    T_val     = 1.0     # Time endowment per consumer
    D0        = 0.5
    epsilon_C = 0.9
    epsilon_D = 0.9
    p_C       = 1.0     # Numeraire
    
    # Production functions:
    F_C = (epsilon_C*(T_c**r) + (1-epsilon_C)*(Z_c**r))**(1/r)
    F_D = (epsilon_D*(T_d**r) + (1-epsilon_D)*(Z_d**r))**(1/r)
    
    # (1) Labor market equilibrium:
    # Representative leisure:
    l = (gamma/((alpha+beta+gamma)*w))*(w*T_val - p_D*D0)
    eq1 = T_c + T_d - (T_val - l)
    
    # (2) Market clearing for good D:
    agg_demand_D = (beta/(p_D*(alpha+beta+gamma)))*(w*T_val - p_D*D0) + D0
    eq2 = agg_demand_D - F_D
    
    # (3) Sector C wage condition:
    eq3 = w - epsilon_C * (T_c**(r-1)) * (F_C**(1-r))
    
    # (4) Sector D wage condition:
    eq4 = w - epsilon_D * (T_d**(r-1)) * (F_D**(1-r)) * p_D
    
    # (5) Sector C optimal input ratio:
    ratio_C = ((1-epsilon_C)/epsilon_C)**(1/(r-1))
    eq5 = Z_c - T_c * ratio_C
    
    # (6) Sector D optimal input ratio:
    ratio_D = ((1-epsilon_D)/epsilon_D)**(1/(r-1))
    eq6 = Z_d - T_d * ratio_D
    
    return np.array([eq1, eq2, eq3, eq4, eq5, eq6])

# --- Initial Guess for x = [T_d, Z_d, T_c, Z_c, w, p_D] ---
x0 = np.array([0.4, 0.4, 0.6, 0.6, 0.5, 1.5])

sol = root(system_eqns, x0, method='hybr')

print("Solution status:", sol.status)
print("Solution message:", sol.message)
print("Solution vector [T_d, Z_d, T_c, Z_c, w, p_D]:")
print(sol.x)

print("\nSolution Summary:")
T_d, Z_d, T_c, Z_c, w, p_D = sol.x
print(f"Sector D: T_prod = {T_d:.4f}, Z_D = {Z_d:.4f}")
print(f"Sector C: T_prod = {T_c:.4f}, Z_C = {Z_c:.4f}")
print(f"Common wage, w = {w:.4f}, p_D = {p_D:.4f}")

# Compute and print the residuals of the FOC and market clearing equations:
residuals = system_eqns(sol.x)
labels = [
    "Labor market equilibrium residual",
    "Good D market clearing residual",
    "Sector C wage condition residual",
    "Sector D wage condition residual",
    "Sector C optimal input ratio residual",
    "Sector D optimal input ratio residual"
]

print("\nResiduals (should be close to zero):")
for label, res in zip(labels, residuals):
    print(f"{label}: {res:.6f}")