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
d0 = 0.5 # Baseline dirty good consumption
epsilon_c = 0.995
epsilon_d = 0.92 # Using the value from the baseline code
p_c = 1.0 # Numeraire
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n = len(phi)

# --- Solve Function (Modified slightly to ensure l_agents is returned) ---
# (Same solve function as the previous version - ensures needed results are returned)
def solve(tau_w, tau_z, g):

    # --- Nested System Equations Function ---
    def system_eqns(y):
        t_c, t_d, log_z_c, log_z_d, w, p_d, l = y
        # Add bounds/checks to prevent log/exponent issues during iteration
        if p_d <= 1e-6 or w <= 1e-6: # Prevent issues with price/wage near zero
            return np.full(7, 1e6)
        z_c = np.exp(min(log_z_c, 20)) # Avoid overflow
        z_d = np.exp(min(log_z_d, 20))

        # Ensure t_c, t_d > 0 for CES
        t_c_eff = max(t_c, 1e-9)
        t_d_eff = max(t_d, 1e-9)

        # Check for potentially very large inputs causing issues
        if t_c_eff > 1e6 or t_d_eff > 1e6 or z_c > 1e6 or z_d > 1e6:
             return np.full(7, 1e6)

        f_c = (epsilon_c * (t_c_eff**r) + (1 - epsilon_c) * (z_c**r))**(1/r)
        f_d = (epsilon_d * (t_d_eff**r) + (1 - epsilon_d) * (z_d**r))**(1/r)

        # Calculate Y_tilde (Full Income net of committed d expenditure potential)
        Y_tilde = phi * w * (1 - tau_w) * t + l - p_d * d0
        if np.any(Y_tilde <= 1e-9): # Check for non-positive income base
             return np.full(7, 1e6)

        # Demands (including leisure calculation)
        d_agents = (beta / (p_d * (alpha + beta + gamma))) * Y_tilde + d0
        # Price of leisure includes net wage
        price_leisure = phi * w * (1 - tau_w)
        if np.any(price_leisure <= 1e-9): # Avoid division by zero
             return np.full(7, 1e6)
        l_agents = (gamma / ((alpha + beta + gamma) * price_leisure)) * Y_tilde
        l_agents = np.clip(l_agents, 1e-9, t - 1e-9) # Ensure leisure is valid

        agg_labor = np.sum(phi * (t - l_agents))
        agg_d = np.sum(d_agents)

        # --- System of Equations ---
        eq1 = t_c + t_d - agg_labor # Labor market clearing
        eq2 = (agg_d + 0.5 * g / p_d) - f_d # Dirty goods market clearing

        # FOCs (using safe t_c_eff, t_d_eff)
        MP_L_c = epsilon_c * (t_c_eff**(r - 1)) * (f_c**(1 - r))
        eq3 = w - MP_L_c
        MP_Z_c = (1 - epsilon_c) * (z_c**(r - 1)) * (f_c**(1 - r))
        eq4 = tau_z - MP_Z_c
        MP_L_d = epsilon_d * (t_d_eff**(r - 1)) * (f_d**(1 - r))
        eq5 = w - MP_L_d * p_d
        MP_Z_d = (1 - epsilon_d) * (z_d**(r - 1)) * (f_d**(1 - r))
        eq6 = tau_z - MP_Z_d * p_d

        # Gov budget constraint
        total_wage_tax_revenue = np.sum(tau_w * w * phi * (t - l_agents))
        total_z_tax_revenue = tau_z * (z_c + z_d)
        eq7 = n * l - (total_wage_tax_revenue + total_z_tax_revenue - g)

        # Check for NaN/Inf in residuals
        residuals = np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7])
        if np.any(np.isnan(residuals)) or np.any(np.isinf(residuals)):
            return np.full(7, 1e7) # Return large penalty if calculation breaks

        return residuals
    # --- End of Nested System Equations Function ---

    # Initial guess
    y0 = np.array([0.3, 0.4, np.log(0.6), np.log(0.4), 0.5, 1.5, 0.1])

    # Solve the system
    sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-8, 'ftol': 1e-8, 'maxiter': 2000})

    # --- Post-processing (extract results needed for analysis) ---
    results = {"sol": sol} # Store solver object
    if sol.success:
        t_c, t_d, log_z_c, log_z_d, w, p_d, l = sol.x
        # Check for extreme values in solution before proceeding
        if not (np.isfinite(sol.x).all() and p_d > 1e-6 and w > 1e-6):
             results.update({
                 "t_d": np.nan, "p_d": np.nan, "d_agents": np.full(n, np.nan),
                 "incomes": np.full(n, np.nan), "l_agents": np.full(n, np.nan) # Added l_agents placeholder
             })
             # Override success status if solution is numerically bad
             sol.success = False
             return results, sol.success


        z_c = np.exp(log_z_c)
        z_d = np.exp(log_z_d)

        # Recalculate key values based on solution
        Y_tilde = phi * w * (1 - tau_w) * t + l - p_d * d0
        # Ensure Y_tilde is positive before proceeding
        if np.any(Y_tilde <= 1e-9):
            results.update({
                "t_d": np.nan, "p_d": np.nan, "d_agents": np.full(n, np.nan),
                "incomes": np.full(n, np.nan), "l_agents": np.full(n, np.nan)
            })
            sol.success = False # Mark as failed if Y_tilde issue arises post-solve
            return results, sol.success


        d_agents = (beta / (p_d * (alpha + beta + gamma))) * Y_tilde + d0
        price_leisure = phi * w * (1 - tau_w)
        l_agents = (gamma / ((alpha + beta + gamma) * price_leisure)) * Y_tilde
        l_agents = np.clip(l_agents, 1e-9, t - 1e-9)
        labor_supply = t - l_agents

        # Calculate income for each agent
        # Income = Net Labor Income + Lump-sum Transfer
        incomes = phi * w * (1 - tau_w) * labor_supply + l

        results.update({
            "t_d": t_d,
            "p_d": p_d,
            "d_agents": d_agents,
            "incomes": incomes,
            "l_agents": l_agents, # Make sure l_agents is available if needed later
        })
    else:
        # Add NaN placeholders if solver fails
        results.update({
            "t_d": np.nan,
            "p_d": np.nan,
            "d_agents": np.full(n, np.nan),
            "incomes": np.full(n, np.nan),
            "l_agents": np.full(n, np.nan) # Add placeholder here too
        })


    return results, sol.success # Return dict and convergence status


# --- Simulation Setup ---

# Fixed parameters
tau_w_fixed = np.array([-1.75, -0.5, 0.0, 0.25, 0.55]) # Klenert optimality values from example
g_fixed = 5.0

# Range for tau_z to iterate over
tau_z_values = np.linspace(0.01, 2.0, 50) # e.g., 50 points from 0.01 to 2.0

# Lists to store results
tau_z_list = []
t_d_list = []
p_d_list = [] # Store p_d
incomes_list = [] # Store incomes arrays
# List to store arrays of total dirty good income shares
total_income_shares_list = []
# List to store arrays of subsistence dirty good income shares
subsistence_shares_list = []


print("Starting simulation for baseline model...")
start_time = time.time()

# --- Main Simulation Loop ---
for tau_z_val in tau_z_values:
    print(f"  Processing tau_z = {tau_z_val:.3f}...")
    try:
        results_dict, converged = solve(tau_w_fixed, tau_z_val, g_fixed)

        tau_z_list.append(tau_z_val) # Store tau_z regardless of convergence

        if converged:
            t_d = results_dict["t_d"]
            p_d = results_dict["p_d"]
            d_agents = results_dict["d_agents"]
            incomes = results_dict["incomes"]

            # Store results needed for both share types
            t_d_list.append(t_d)
            p_d_list.append(p_d)
            incomes_list.append(incomes) # Store the array of incomes

            # Calculate Total dirty good income share
            total_dirty_expenditure = p_d * d_agents
            total_income_shares = np.divide(total_dirty_expenditure, incomes,
                                            out=np.full_like(incomes, np.nan),
                                            where=incomes > 1e-9)
            total_income_shares_list.append(total_income_shares)

            # Calculate Subsistence dirty good income share
            subsistence_expenditure = p_d * d0
            subsistence_shares = np.divide(subsistence_expenditure, incomes,
                                           out=np.full_like(incomes, np.nan),
                                           where=incomes > 1e-9)
            subsistence_shares_list.append(subsistence_shares)


        else:
            # Append NaNs if solver failed
            t_d_list.append(np.nan)
            p_d_list.append(np.nan)
            incomes_list.append(np.full(n, np.nan))
            total_income_shares_list.append(np.full(n, np.nan))
            subsistence_shares_list.append(np.full(n, np.nan))
            print(f"    WARNING: Solver failed for tau_z = {tau_z_val:.3f}")
            # Optional: print solver message results_dict['sol'].message

    except Exception as e:
        # Handle potential errors during solving or calculation
        print(f"    ERROR: Exception occurred for tau_z = {tau_z_val:.3f}: {e}")
        # Append NaNs safely
        if len(tau_z_list) > len(t_d_list): # Ensure lists stay aligned if error before appends
            t_d_list.append(np.nan)
            p_d_list.append(np.nan)
            incomes_list.append(np.full(n, np.nan))
            total_income_shares_list.append(np.full(n, np.nan))
            subsistence_shares_list.append(np.full(n, np.nan))


end_time = time.time()
print(f"Simulation finished in {end_time - start_time:.2f} seconds.")

# Convert lists to NumPy arrays for easier plotting
tau_z_array = np.array(tau_z_list)
t_d_array = np.array(t_d_list)
p_d_array = np.array(p_d_list) # Added
incomes_array = np.array(incomes_list) # Added, shape (num_steps, n)
total_income_shares_array = np.array(total_income_shares_list) # shape (num_steps, n)
subsistence_shares_array = np.array(subsistence_shares_list) # Added, shape (num_steps, n)


# --- Plotting Results ---
print("Plotting results...")

# Plot 1: t_d vs tau_z (No change here)
fig1, ax1 = plt.subplots(figsize=(8, 5))
valid_td_mask = ~np.isnan(t_d_array) # Mask for converged points
ax1.plot(tau_z_array[valid_td_mask], t_d_array[valid_td_mask], marker='.', linestyle='-')
ax1.set_xlabel("Tax on Z input ($\\tau_z$)")
ax1.set_ylabel("Equilibrium Labor in Dirty Sector ($t_d$)")
ax1.set_title("Dirty Sector Labor Input vs. Z-Tax (Baseline Model)")
ax1.grid(True)
plt.tight_layout()

# Plot 2: Income Shares vs tau_z (Modified)
fig2, ax2 = plt.subplots(figsize=(12, 7)) # Make figure slightly wider for legend
for i in range(n): # Loop through each household type
    # Get total income shares for household i
    total_shares_hh_i = total_income_shares_array[:, i]
    # Get subsistence income shares for household i
    subsistence_shares_hh_i = subsistence_shares_array[:, i]

    # Use a single mask based on total shares (assuming failure affects both)
    valid_mask = ~np.isnan(total_shares_hh_i)

    # Plot Total Share (Solid Line)
    ax2.plot(tau_z_array[valid_mask], total_shares_hh_i[valid_mask],
             marker='.', linestyle='-', color=f'C{i}', # Assign color based on index
             label=f'HH {i+1} Total Share ($\\phi$={phi[i]:.3f})')

    # Plot Subsistence Share (Dashed Line, same color)
    ax2.plot(tau_z_array[valid_mask], subsistence_shares_hh_i[valid_mask],
             marker=None, linestyle='--', color=f'C{i}', # Use same color
             label=f'HH {i+1} Subsist. Share') # Simplified label


ax2.set_xlabel("Tax on Z input ($\\tau_z$)")
ax2.set_ylabel("Dirty Good Expenditure / Total Income")
ax2.set_title("Income Share Spent on Dirty Good vs. Z-Tax (Baseline Model)")

# Improve legend handling
handles, labels = ax2.get_legend_handles_labels()
# Create a clearer legend - could group by household if needed, but this is simpler
ax2.legend(handles, labels, title="Household Type & Share", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

ax2.grid(True, linestyle=':') # Lighter grid
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y))) # Format y-axis as percentage
plt.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust layout more for potentially longer legend

plt.show()

print("Script finished.")