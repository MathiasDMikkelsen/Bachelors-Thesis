import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import time # Optional: to time the execution

# --- Parameters ---
alpha = 0.7
beta = 0.2
gamma = 0.2
r = -1.0
t = 24.0
# d0 = 0.5  # d0 will now be passed as an argument to solve
epsilon_c = 0.995
epsilon_d = 0.582302 # NOTE: Using the value from the user's code block
p_c = 1.0 # Numeraire

# Productivity weights (already summing to 1 within each sector in user's code)
phi_d = np.array([0.053164, 0.322405, (1-0.053164-0.322405)])  # Dirty sector households
phi_c = np.array([0.295018, (1-0.295018)])                     # Clean sector households

# Combine and get counts
phi = np.concatenate([phi_d, phi_c])
n = len(phi) # Total number of household types (5)
n_d = len(phi_d) # Number of dirty types (3)
n_c = len(phi_c) # Number of clean types (2)

# --- Modified Solve Function ---
# Now accepts d0 as an argument
def solve(tau_w, tau_z, g, d0):

    # Ensure tau_w matches the structure of phi
    if len(tau_w) != n:
        raise ValueError(f"Length of tau_w ({len(tau_w)}) must match total number of households ({n})")

    # --- Nested System Equations Function ---
    # This function now uses the d0 passed to the outer solve function
    def system_eqns(y):
        # Unpack variables
        t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = y
        z_c = np.exp(log_z_c)
        z_d = np.exp(log_z_d)

        # --- Firm side ---
        f_c = (epsilon_c * ((t_c + 1e-9)**r) + (1 - epsilon_c) * ((z_c + 1e-9)**r))**(1/r)
        f_d = (epsilon_d * ((t_d + 1e-9)**r) + (1 - epsilon_d) * ((z_d + 1e-9)**r))**(1/r)

        # --- Household side ---
        wage = np.concatenate([w_d * np.ones(n_d), w_c * np.ones(n_c)])
        # Use the passed d0 here
        Y_tilde = phi * wage * (1 - tau_w) * t + l - p_d * d0

        if np.any(Y_tilde <= 0):
            return np.full(8, 1e6) # Return large residuals if Y_tilde non-positive

        denom_shares = alpha + beta + gamma
        # Use the passed d0 here
        d_agents = (beta / (p_d * denom_shares)) * Y_tilde + d0
        price_leisure = phi * wage * (1 - tau_w)
        l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_tilde
        l_agents = np.clip(l_agents, 1e-9, t - 1e-9)
        labor_supply_agents = t - l_agents

        # --- Aggregation ---
        agg_labor_d = np.sum(phi_d * labor_supply_agents[:n_d])
        agg_labor_c = np.sum(phi_c * labor_supply_agents[n_d:])
        agg_d = np.sum(d_agents)

        # --- System of Equations ---
        eq1 = t_c - agg_labor_c # Labor market clearing (clean)
        eq2 = t_d - agg_labor_d # Labor market clearing (dirty)
        eq3 = (agg_d + 0.5 * g / (p_d + 1e-9)) - f_d # Goods market clearing (dirty)

        # Firm FOCs
        MP_L_c = epsilon_c * ((t_c + 1e-9)**(r - 1)) * (f_c**(1 - r))
        eq4 = w_c - MP_L_c
        MP_Z_c = (1 - epsilon_c) * ((z_c + 1e-9)**(r - 1)) * (f_c**(1 - r))
        eq5 = tau_z - MP_Z_c
        MP_L_d = epsilon_d * ((t_d + 1e-9)**(r - 1)) * (f_d**(1 - r))
        eq6 = w_d - MP_L_d * p_d
        MP_Z_d = (1 - epsilon_d) * ((z_d + 1e-9)**(r - 1)) * (f_d**(1 - r))
        eq7 = tau_z - MP_Z_d * p_d

        # Government budget constraint
        total_wage_tax_revenue = np.sum(tau_w * phi * wage * labor_supply_agents)
        total_z_tax_revenue = tau_z * (z_c + z_d)
        eq8 = n * l - (total_wage_tax_revenue + total_z_tax_revenue - g)

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])
    # --- End of Nested System Equations Function ---

    # Initial guess (can sometimes be sensitive)
    # Use a slightly more robust initial guess if needed
    y0 = np.array([5.0, 5.0, np.log(0.6), np.log(0.4), 0.5, 0.6, 1.5, 0.1])

    # Solve the system
    sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-8, 'ftol': 1e-8, 'maxiter': 1000}) # Increased maxiter

    # --- Post-processing (extract t_d) ---
    if sol.success:
        t_d_solution = sol.x[1] # t_d is the second element
    else:
        t_d_solution = np.nan # Indicate failure to converge

    # Return only t_d and convergence status for this analysis
    return t_d_solution, sol.success

# --- Simulation Setup ---

# Fixed parameters for the simulation runs
tau_w_fixed = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
g_fixed = 5.0

# Ranges for variables to loop over
d0_values = np.arange(0.1, 2.01, 0.1) # From 0.1 to 2.0, step 0.1
tau_z_values = np.linspace(0.01, 1.5, 30) # Range for tau_z (e.g., 30 points from 0.01 to 1.5)

# Dictionary to store results
# Format: {d0_value: {'tau_z': [list of tau_z], 't_d': [list of t_d]}}
results_by_d0 = {}

print("Starting simulation...")
start_time = time.time()

# --- Main Simulation Loop ---
for d0_val in d0_values:
    print(f"  Processing d0 = {d0_val:.1f}...")
    current_tau_z_list = []
    current_t_d_list = []

    for tau_z_val in tau_z_values:
        try:
            # Solve the model for the current d0 and tau_z
            t_d_result, converged = solve(tau_w_fixed, tau_z_val, g_fixed, d0_val)

            if converged:
                current_tau_z_list.append(tau_z_val)
                current_t_d_list.append(t_d_result)
            else:
                # Store NaN if not converged to avoid gaps in plots but show failure
                current_tau_z_list.append(tau_z_val)
                current_t_d_list.append(np.nan)
                print(f"    WARNING: Solver failed for d0={d0_val:.1f}, tau_z={tau_z_val:.3f}")

        except Exception as e:
            # Handle potential errors during solving (e.g., invalid values)
            current_tau_z_list.append(tau_z_val)
            current_t_d_list.append(np.nan)
            print(f"    ERROR: Exception occurred for d0={d0_val:.1f}, tau_z={tau_z_val:.3f}: {e}")

    # Store results for this d0 value
    results_by_d0[round(d0_val, 2)] = { # Use rounded key for robustness
        'tau_z': np.array(current_tau_z_list),
        't_d': np.array(current_t_d_list)
    }

end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

# --- Plotting Results ---
print("Plotting results...")
fig, ax = plt.subplots(figsize=(10, 6))

for d0_val, data in results_by_d0.items():
    # Plot only if there are valid (non-NaN) data points
    if np.any(~np.isnan(data['t_d'])):
         # Use masking to handle potential NaNs gracefully in plotting
        valid_mask = ~np.isnan(data['t_d'])
        ax.plot(data['tau_z'][valid_mask], data['t_d'][valid_mask],
                marker='.', linestyle='-', label=f'd0 = {d0_val:.1f}')

ax.set_xlabel("Tax on Z input ($\\tau_z$)")
ax.set_ylabel("Equilibrium Labor in Dirty Sector ($t_d$)")
ax.set_title("Dirty Sector Labor Input vs. Z-Tax for Different Baseline Demands ($d_0$)")
ax.legend(title="$d_0$ values", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)
plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
plt.show()

print("Script finished.")