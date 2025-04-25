import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
import pandas as pd
import traceback # Import for detailed error printing

# a. parameters (As defined in your script)
alpha = 0.7
beta = 0.2
gamma = 0.2
r = -1.0
t = 24.0
d0 = 0.5
epsilon_c = 0.995
epsilon_d = 0.92
p_c = 1.0
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n = len(phi)

# --- solve function (With improved stability/error handling) ---
def solve(tau_w, tau_z, g, y0_guess=None):

    # --- Nested system_eqns function ---
    def system_eqns(y):
        t_c, t_d, log_z_c, log_z_d, w, p_d, l = y

        # --- Early checks for invalid inputs from solver ---
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            # print("Warning: NaN/Inf input to system_eqns")
            return np.full(7, 1e12) # Return large error if input is bad

        # Ensure inputs result in positive values where needed
        # Add small constant before log to handle potential zeros during iteration
        z_c = np.exp(log_z_c); z_d = np.exp(log_z_d)
        t_c_eff = max(t_c, 1e-9); t_d_eff = max(t_d, 1e-9) # Ensure positive inputs for powers
        z_c_eff = max(z_c, 1e-9); z_d_eff = max(z_d, 1e-9)
        p_d_eff = max(p_d, 1e-9) # Ensure positive price
        w_eff = max(w, 1e-9) # Ensure positive wage

        # Check if l (transfer) is reasonable, maybe bound it? Large negative l could cause issues.
        # if l < -100: # Arbitrary bound example
        #    return np.full(7, 1e6)

        # --- Calculations ---
        try:
            f_c = (epsilon_c * (t_c_eff**r) + (1 - epsilon_c) * (z_c_eff**r))**(1/r)
            f_d = (epsilon_d * (t_d_eff**r) + (1 - epsilon_d) * (z_d_eff**r))**(1/r)

            Y_term = phi * w_eff * (1 - tau_w) * t + l - p_d_eff * d0
            if np.any(Y_term <= 1e-9): return np.full(7, 1e6) # Needs positive Y_term for logs in utility

            denom_shares = alpha + beta + gamma
            d_agents = (beta / (p_d_eff * denom_shares)) * Y_term + d0
            price_leisure = phi * w_eff * (1 - tau_w)
            if np.any(price_leisure <= 1e-9): return np.full(7, 1e6) # Check price_leisure again
            l_agents = (gamma / (denom_shares * price_leisure)) * Y_term
            l_agents = np.clip(l_agents, 1e-9, t - 1e-9) # Clip leisure to be within [eps, t-eps]
            labor_supply_agents = t - l_agents

            agg_labor = np.sum(phi * labor_supply_agents)
            agg_d = np.sum(d_agents)

            MP_L_c = epsilon_c * (t_c_eff**(r - 1)) * (f_c**(1 - r));
            MP_Z_c = (1 - epsilon_c) * (z_c_eff**(r - 1)) * (f_c**(1 - r));
            MP_L_d = epsilon_d * (t_d_eff**(r - 1)) * (f_d**(1 - r));
            MP_Z_d = (1 - epsilon_d) * (z_d_eff**(r - 1)) * (f_d**(1 - r));

            # --- Equations ---
            eq1 = t_c + t_d - agg_labor
            eq2 = (agg_d + 0.5 * g / p_d_eff) - f_d
            eq3 = w - MP_L_c
            eq4 = tau_z - MP_Z_c
            eq5 = w - MP_L_d * p_d
            eq6 = tau_z - MP_Z_d * p_d
            total_wage_tax_revenue = np.sum(tau_w * w * phi * labor_supply_agents)
            total_z_tax_revenue = tau_z * (z_c + z_d) # Use original z_c, z_d here
            eq7 = n * l - (total_wage_tax_revenue + total_z_tax_revenue - g)

            eqs = np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7])
            # Check if equations result in NaN/Inf
            if np.any(np.isnan(eqs)) or np.any(np.isinf(eqs)):
                # print("Warning: NaN/Inf in equations")
                return np.full(7, 1e12)
            return eqs

        except (ValueError, OverflowError, ZeroDivisionError) as e:
            # Catch potential math errors during calculation
            # print(f"Warning: Math error in system_eqns - {e}")
            return np.full(7, 1e12) # Return large error

    # --- End of nested function ---

    if y0_guess is None:
        y0 = np.array([0.3, 0.4, np.log(0.6), np.log(0.4), 0.5, 1.5, 0.1])
    else:
        # Check if the guess itself contains NaN/Inf from previous failure
        if np.any(np.isnan(y0_guess)) or np.any(np.isinf(y0_guess)):
             print("Warning: Bad initial guess provided, resetting to default y0.")
             y0 = np.array([0.3, 0.4, np.log(0.6), np.log(0.4), 0.5, 1.5, 0.1])
        else:
             y0 = y0_guess

    # Solve the system - Consider trying 'hybr' if 'lm' fails often
    try:
        sol = root(system_eqns, y0, method='lm', options={'xtol': 1e-9, 'ftol': 1e-9, 'maxiter': 1000})
        # sol = root(system_eqns, y0, method='hybr', options={'xtol': 1e-9}) # Alternative solver

    except Exception as e:
        print(f"ERROR: root finding failed with exception: {e}")
        # Create a dummy failed sol object
        from scipy.optimize.optimize import OptimizeResult
        sol = OptimizeResult(x=np.full(len(y0), np.nan), success=False, status=-1, message=f"Exception: {e}")


    # --- Post-processing ---
    if not sol.success:
        nan_results = {k: np.nan for k in ["t_c", "t_d", "z_c", "z_d", "w", "p_d", "l", "f_c", "f_d", "agg_c", "agg_d", "agg_labor", "profit_c", "profit_d"]}
        nan_results.update({k: np.full(n, np.nan) for k in ["c_agents", "d_agents", "l_agents", "labor_supply_agents", "budget_errors", "utilities"]})
        nan_results["sol"] = sol
        return np.full(len(y0), np.nan), nan_results, False

    # If successful, proceed with calculations
    t_c, t_d, log_z_c, log_z_d, w, p_d, l = sol.x
    z_c = np.exp(log_z_c); z_d = np.exp(log_z_d)

    # Use try-except for post-processing calculations as well
    try:
        f_c = (epsilon_c * (t_c**r) + (1 - epsilon_c) * (z_c**r))**(1/r)
        f_d = (epsilon_d * (t_d**r) + (1 - epsilon_d) * (z_d**r))**(1/r)

        Y_term = phi * w * (1 - tau_w) * t + l - p_d * d0
        denom_shares = alpha + beta + gamma
        c_agents = (alpha / (p_c * denom_shares)) * Y_term
        d_agents = (beta / ((p_d + 1e-9) * denom_shares)) * Y_term + d0
        price_leisure = phi * w * (1 - tau_w)
        l_agents = (gamma / (denom_shares * (price_leisure + 1e-9))) * Y_term
        l_agents = np.clip(l_agents, 1e-9, t - 1e-9)
        labor_supply_agents = t - l_agents

        agg_c = np.sum(c_agents); agg_d = np.sum(d_agents)
        agg_labor = np.sum(phi * labor_supply_agents)

        profit_c = f_c - w * t_c - tau_z * z_c
        profit_d = p_d * f_d - w * t_d - tau_z * z_d

        budget_errors = np.zeros(n)
        for i in range(n):
            income = phi[i] * w * (1 - tau_w[i]) * labor_supply_agents[i] + l
            expenditure = c_agents[i] + p_d * d_agents[i]
            budget_errors[i] = income - expenditure

        utilities = np.zeros(n)
        for i in range(n):
            if c_agents[i] > 1e-9 and (d_agents[i] - d0) > 1e-9 and l_agents[i] > 1e-9:
                utilities[i] = alpha * np.log(c_agents[i]) + beta * np.log(d_agents[i] - d0) + gamma * np.log(l_agents[i])
            else: utilities[i] = -np.inf

        residuals = system_eqns(sol.x) # Recalculate residuals

        results = {
            "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d, "w": w, "p_d": p_d, "l": l,
            "f_c": f_c, "f_d": f_d,
            "c_agents": c_agents, "d_agents": d_agents, "l_agents": l_agents,
            "labor_supply_agents": labor_supply_agents, # Added
            "agg_c": agg_c, "agg_d": agg_d, "agg_labor": agg_labor,
            "profit_c": profit_c, "profit_d": profit_d,
            "budget_errors": budget_errors,
            "utilities": utilities,
            "residuals": residuals, # Added residuals
            "sol": sol
        }
        return sol.x, results, sol.success

    except (ValueError, OverflowError, ZeroDivisionError, RuntimeWarning) as e:
        print(f"ERROR: Post-processing failed for tau_z={tau_z} with exception: {e}")
        # Return failure if post-processing calculations fail
        nan_results = {k: np.nan for k in ["t_c", "t_d", "z_c", "z_d", "w", "p_d", "l", "f_c", "f_d", "agg_c", "agg_d", "agg_labor", "profit_c", "profit_d"]}
        nan_results.update({k: np.full(n, np.nan) for k in ["c_agents", "d_agents", "l_agents", "labor_supply_agents", "budget_errors", "utilities"]})
        nan_results["sol"] = sol # Keep original sol object
        nan_results["residuals"] = np.full(7, np.nan)
        return sol.x, nan_results, False # Return success=False but keep original solution vector


# --- Single run section (kept for reference) ---
# (You can comment this out if you only want the loop)
# ... (single run code from previous snippet) ...


# --- Analysis Loop Section ---
print("\n" + "="*50)
print("Running analysis over a range of tau_z")
print("="*50)

tau_z_range = np.linspace(0.1, 2.0, 40)
tau_w_loop = np.array([0.015, 0.072, 0.115, 0.156, 0.24]) # Use consistent tau_w
g_loop = 5.0
results_list = []
tau_z_values = []
last_successful_y = None
default_y0 = np.array([0.3, 0.4, np.log(0.6), np.log(0.4), 0.5, 1.5, 0.1]) # Keep default

for tau_z_val in tau_z_range:
    print(f"Solving for tau_z = {tau_z_val:.3f} ... ", end="")
    # Use default y0 if last_successful_y is None (e.g., first run or after failure)
    current_guess = last_successful_y if last_successful_y is not None else default_y0
    try:
        solution_vec, results_dict, converged_flag = solve(tau_w_loop, tau_z_val, g_loop, y0_guess=current_guess)

        # Store basic info regardless of convergence
        results_dict['tau_z'] = tau_z_val
        results_dict['converged'] = converged_flag
        results_list.append(results_dict)
        tau_z_values.append(tau_z_val) # Keep track of all attempted tau_z

        if converged_flag:
            print(f"Converged. t_d = {results_dict.get('t_d', np.nan):.4f}")
            last_successful_y = solution_vec # Update guess ONLY if successful
        else:
            print(f"FAILED. Solver status: {results_dict['sol'].status}, Msg: {results_dict['sol'].message}")
            last_successful_y = None # Reset guess after failure


    except Exception as e:
        print(f"ERROR during solve call for tau_z={tau_z_val}: {e}")
        traceback.print_exc() # Print full traceback
        # Store minimal info indicating error
        error_results = {'tau_z': tau_z_val, 'converged': False, 't_d': np.nan}
        results_list.append(error_results)
        tau_z_values.append(tau_z_val)
        last_successful_y = None # Reset guess after error


# --- Process and Display Results ---
if results_list:
    df = pd.DataFrame(results_list)
    # Ensure ratio calculation handles potential NaN/zero in z_d
    df['z_d'] = df['z_d'].replace(0, np.nan) # Avoid zero division
    df['t_d/z_d_ratio'] = df['t_d'] / df['z_d']

    display_cols = ['tau_z', 'converged', 't_d', 't_c', 'z_d', 'z_c', 'w', 'p_d', 'l', 'f_d', 'agg_d', 'agg_labor', 't_d/z_d_ratio']
    df_display = pd.DataFrame({col: df.get(col, np.nan) for col in display_cols})

    print("\n" + "="*60)
    print("Simulation Results Summary (Homogenous Model vs. tau_z)")
    print("="*60)
    with pd.option_context('display.max_rows', None, 'display.precision', 4):
        print(df_display)
    print("="*60)

    # --- Plotting ---
    converged_df = df[df['converged']].copy() # Plot only converged points

    # <<< Add check here before plotting >>>
    print(f"\nNumber of converged points found: {len(converged_df)}")
    if converged_df.empty:
        print("No converged points available to plot. Check solver messages and parameters.")
    else:
        print("Attempting to generate plots...")
        try:
            fig, axes = plt.subplots(3, 2, figsize=(12, 15))
            fig.suptitle('Homogenous Model Analysis vs. tau_z', fontsize=16)

            # Plot t_d
            axes[0, 0].plot(converged_df['tau_z'], converged_df['t_d'], marker='o', linestyle='-')
            axes[0, 0].set_title('Labor in Dirty Sector (t_d)')
            axes[0, 0].set_xlabel('tau_z')
            axes[0, 0].set_ylabel('t_d')
            axes[0, 0].grid(True)

            # Plot z_d
            axes[0, 1].plot(converged_df['tau_z'], converged_df['z_d'], marker='o', linestyle='-')
            axes[0, 1].set_title('Input z in Dirty Sector (z_d)')
            axes[0, 1].set_xlabel('tau_z')
            axes[0, 1].set_ylabel('z_d')
            axes[0, 1].grid(True)

            # Plot t_d / z_d ratio (Substitution)
            axes[1, 0].plot(converged_df['tau_z'], converged_df['t_d/z_d_ratio'], marker='o', linestyle='-')
            axes[1, 0].set_title('Labor/Input Ratio in Dirty Sector (t_d / z_d)')
            axes[1, 0].set_xlabel('tau_z')
            axes[1, 0].set_ylabel('t_d / z_d')
            axes[1, 0].grid(True)
            axes[1, 0].set_ylim(bottom=0) # Ratio likely non-negative

            # Plot f_d (Output/Scale)
            axes[1, 1].plot(converged_df['tau_z'], converged_df['f_d'], marker='o', linestyle='-')
            axes[1, 1].set_title('Output of Dirty Sector (f_d)')
            axes[1, 1].set_xlabel('tau_z')
            axes[1, 1].set_ylabel('f_d')
            axes[1, 1].grid(True)

            # Plot p_d (Relative Price)
            axes[2, 0].plot(converged_df['tau_z'], converged_df['p_d'], marker='o', linestyle='-')
            axes[2, 0].set_title('Price of Dirty Good (p_d)')
            axes[2, 0].set_xlabel('tau_z')
            axes[2, 0].set_ylabel('p_d')
            axes[2, 0].grid(True)

            # Plot w (Wage)
            axes[2, 1].plot(converged_df['tau_z'], converged_df['w'], marker='o', linestyle='-')
            axes[2, 1].set_title('Economy-wide Wage (w)')
            axes[2, 1].set_xlabel('tau_z')
            axes[2, 1].set_ylabel('w')
            axes[2, 1].grid(True)

            plt.tight_layout(rect=[0, 0.03, 1, 0.96])
            plt.show()
        except Exception as plot_err:
            print(f"\nAn error occurred during plotting: {plot_err}")
            traceback.print_exc()


else:
    print("\nNo results were generated from the loop.")