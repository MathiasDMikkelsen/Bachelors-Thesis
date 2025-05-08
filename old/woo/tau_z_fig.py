import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt

# Global model parameters (from your LaTeX)
alpha     = 0.7
beta      = 0.2
gamma     = 0.2
r         = -1.0
T_val     = 1000.0      # Time endowment per household
L         = 0.0       # Lump-sum transfers
D0        = 0.5
epsilon_C = 0.995     # Example value for Sector C
epsilon_D = 0.92      # Example value for Sector D
p_C       = 1.0       # Numeraire

# tau_z is a given parameter; we will vary it.
tau_z = 1.0  # initial value, will be overwritten in loop

# Household weights (for 5 households, each is set so that sum(phi)=1)
# In your model these are given by:
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.511])
# (Note: these already sum to 1, so aggregate T = 1)
n = len(phi)

def system_eqns(x):
    # Unpack unknowns:
    # x = [T_C, T_D, Z_C, Z_D, w, p_D]
    T_C, T_D, Z_C, Z_D, w, p_D = x

    F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
    F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)
    
    # Household-level demands (as in your LaTeX)
    C_agents = (alpha/(p_C*(alpha+beta+gamma)))*(phi*w*T_val + L - p_D*D0)
    D_agents = (beta/(p_D*(alpha+beta+gamma)))*(phi*w*T_val + L - p_D*D0) + D0
    l_agents = (gamma/((alpha+beta+gamma)*phi*w))*(phi*w*T_val + L - p_D*D0)

    # Aggregate outcomes:
    agg_labor = np.sum(phi*(T_val - l_agents))  # total labor supply
    agg_D = np.sum(D_agents)                      # aggregate demand for good D

    # Equilibrium conditions:
    # (1) Labor market equilibrium:
    eq1 = T_C + T_D - agg_labor

    # (2) Market clearing for good D:
    eq2 = agg_D - F_D

    # Production FOCs:
    # (3) Sector C wage FOC:
    eq3 = w - epsilon_C * (T_C**(r-1)) * (F_C**(1-r))
    # (4) Sector C margin FOC (τ_z is given):
    eq4 = tau_z - (1 - epsilon_C) * (Z_C**(r-1)) * (F_C**(1-r))
    
    # (5) Sector D wage FOC:
    eq5 = w - epsilon_D * (T_D**(r-1)) * (F_D**(1-r)) * p_D
    # (6) Sector D margin FOC:
    eq6 = tau_z - (1 - epsilon_D) * (Z_D**(r-1)) * (F_D**(1-r)) * p_D

    return np.array([eq1, eq2, eq3, eq4, eq5, eq6])

# Initial guess for x = [T_C, T_D, Z_C, Z_D, w, p_D]
x0 = np.array([0.3, 0.4, 0.6, 0.4, 0.5, 1.5])

# Define a range for tau_z to vary:
tau_range = np.linspace(0.001, 1.0, 1000)

# Prepare arrays to store solutions
sol_array = np.empty((len(tau_range), 6))
sol_array[:] = np.nan

for i, tau in enumerate(tau_range):
    # Set global tau_z to current value
    # (Since system_eqns uses the global variable tau_z, we update it here.)
    tau_z = tau
    sol = root(system_eqns, x0, method='hybr')
    if sol.success:
        sol_array[i, :] = sol.x
    else:
        print(f"Solution did not converge for tau_z = {tau}")

# Unpack solution arrays (columns correspond to T_C, T_D, Z_C, Z_D, w, p_D)
T_C_vals = sol_array[:, 0]
T_D_vals = sol_array[:, 1]
Z_C_vals = sol_array[:, 2]
Z_D_vals = sol_array[:, 3]
w_vals   = sol_array[:, 4]
p_D_vals = sol_array[:, 5]

# Plot the solution variables as a function of tau_z:
plt.figure(figsize=(10, 8))

plt.subplot(3, 2, 1)
plt.plot(tau_range, T_C_vals, '-')
plt.xlabel('tau_z')
plt.ylabel('T_C')

plt.subplot(3, 2, 2)
plt.plot(tau_range, T_D_vals, '-')
plt.xlabel('tau_z')
plt.ylabel('T_D')

plt.subplot(3, 2, 3)
plt.plot(tau_range, Z_C_vals, '-')
plt.xlabel('tau_z')
plt.ylabel('Z_C')

plt.subplot(3, 2, 4)
plt.plot(tau_range, Z_D_vals, '-')
plt.xlabel('tau_z')
plt.ylabel('Z_D')

plt.subplot(3, 2, 5)
plt.plot(tau_range, w_vals, '-')
plt.xlabel('tau_z')
plt.ylabel('w')

plt.subplot(3, 2, 6)
plt.plot(tau_range, p_D_vals, '-')
plt.xlabel('tau_z')
plt.ylabel('p_D')

plt.tight_layout()
plt.show()

# Additionally, you might want to compute and print residuals for one chosen tau_z.
# For example, for the last tau_z value:
tau_z = tau_range[-1]
final_residuals = system_eqns(sol_array[-1, :])
labels = [
    "Labor market equilibrium residual",
    "Good D market clearing residual",
    "Sector C wage FOC residual",
    "Sector C margin FOC residual",
    "Sector D wage FOC residual",
    "Sector D margin FOC residual"
]
print("\nFinal tau_z =", tau_z)
for label, res in zip(labels, final_residuals):
    print(f"{label}: {res:.10f}")

# Also, for each household, compute household demands for the last tau_z:
C_agents = (alpha/(p_C*(alpha+beta+gamma)))*(phi*w_vals[-1]*T_val + L - p_D_vals[-1]*D0)
D_agents = (beta/(p_D_vals[-1]*(alpha+beta+gamma)))*(phi*w_vals[-1]*T_val + L - p_D_vals[-1]*D0) + D0
l_agents = (gamma/((alpha+beta+gamma)*phi*w_vals[-1]))*(phi*w_vals[-1]*T_val + L - p_D_vals[-1]*D0)

print("\nHousehold Equilibrium Outcomes for last tau_z value:")
for i in range(n):
    print(f"Household {i+1}: C = {C_agents[i]:.4f}, D = {D_agents[i]:.4f}, l = {l_agents[i]:.4f}")