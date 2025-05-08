import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
# Ensure the inner solver module name matches your file name (e.g., inner_labor.py)
import inner_labor as solver
# Import necessary parameters from the inner solver
# --- MODIFIED: Added n_d import ---
from inner_labor import alpha, beta, gamma, d0, phi, n, n_d

# a. parameters
T = 24      # Assuming T=24 is the correct time endowment
theta = 1.0
G = 1.0
# --- ADDED: Calculate n_c ---
n_c = n - n_d
# --- ADDED: Calculate number of within-market IC constraints ---
num_constraints = n_d * (n_d - 1) + n_c * (n_c - 1)

# b. maximize social welfare
def maximize_welfare(G, xi):

    # b1. define objective function (remains the same)
    def swf_obj(x):
        tau_w = x[0:n]
        tau_z = x[n]
        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return 1e10
            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d']
            welfare = np.sum(utilities) - 5 * xi * (agg_polluting ** theta)
            return -welfare
        except Exception as e:
            print(f"Solver failed during objective evaluation: {e}")
            return 1e10

    # b2. define SEPARATE IC constraints for each labor market
    def ic_constraints(x):
        tau_w = x[0:n]
        tau_z = x[n]

        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                # Return large negative values, size matches NEW number of constraints
                return -np.ones(num_constraints) * 1e6 # Use updated num_constraints

            # Pre-calculate terms
            I = np.zeros(n)
            wage_vector = results['wage_vector']
            l_agents = results['l_agents']
            for j in range(n):
                labor_supply_j = T - l_agents[j]
                I[j] = labor_supply_j * (1.0 - tau_w[j]) * phi[j] * wage_vector[j]

            g_list = [] # To store constraint values U_i - U_i_j
            utilities = results['utilities']
            c_agents = results['c_agents']
            d_agents = results['d_agents']

            for i in range(n):
                is_i_dirty = (i < n_d) # Check if agent i is in dirty sector
                U_i = utilities[i]
                if np.isinf(U_i): # Skip agent i if their own utility is invalid
                    # Need to add placeholder values if we skip to maintain array size
                    # This case shouldn't happen if inner solver converges properly
                    print(f"Warning: Agent {i} has infinite utility. Skipping IC checks for this agent.")
                    num_skipped_for_i = n_d - 1 if is_i_dirty else n_c - 1
                    g_list.extend([1e6] * num_skipped_for_i) # Add placeholders assuming constraints satisfied
                    continue


                for j in range(n):
                    if i == j:
                        continue

                    is_j_dirty = (j < n_d) # Check if agent j is in dirty sector

                    # --- MODIFIED: Check if i and j are in the SAME labor market ---
                    if (is_i_dirty and is_j_dirty) or (not is_i_dirty and not is_j_dirty):
                        # --- Perform mimicking calculation ONLY if in the same market ---
                        c_j = c_agents[j]
                        d_j = d_agents[j]

                        denom = (1.0 - tau_w[j]) * phi[i] * wage_vector[i]
                        if denom <= 1e-9:
                            g_list.append(1e6) # Constraint satisfied if cannot mimic
                            continue

                        ell_i_j = T - I[j] / denom

                        U_i_j = -np.inf
                        if ell_i_j > 1e-9 and ell_i_j < T - 1e-9 and c_j > 1e-9 and d_j > d0 + 1e-9:
                            try:
                                log_c_j = np.log(c_j)
                                log_d_j_net = np.log(d_j - d0)
                                log_ell_i_j = np.log(ell_i_j)
                                U_i_j = (alpha * log_c_j + beta * log_d_j_net + gamma * log_ell_i_j)
                            except ValueError:
                                U_i_j = -np.inf

                        constraint_value = U_i - U_i_j
                        if np.isinf(constraint_value): # Handles U_i - (-inf)
                            g_list.append(1e6)
                        else:
                            g_list.append(constraint_value)
                    # --- End of check for same labor market ---

            # Check if the number of constraints generated matches expectation
            if len(g_list) != num_constraints:
                 print(f"Warning: Number of IC constraints generated ({len(g_list)}) does not match expected ({num_constraints}). Check for infinite utilities or logic errors.")
                 # Pad with large values if too short, or truncate if too long (though padding is safer)
                 g_list.extend([1e6] * (num_constraints - len(g_list)))
                 g_list = g_list[:num_constraints]


            return np.array(g_list)

        except Exception as e:
            print(f"ic constraint evaluation failed with error: {e}")
            # Return large negative values, size matches NEW number of constraints
            return -np.ones(num_constraints) * 1e6 # Use updated num_constraints

    # b3. set up the nonlinear constraint for the IC conditions
    # --- MODIFIED: Use updated num_constraints ---
    print(f"Setting up {num_constraints} IC constraints ({n_d*(n_d-1)} dirty, {n_c*(n_c-1)} clean).")
    nonlinear_constraint = NonlinearConstraint(ic_constraints, lb=np.zeros(num_constraints), ub=np.inf)

    # b4. initial guess (remains the same)
    initial_tau_w = [0.0] * n
    initial_tau_z = 0.5
    initial_guess = np.array(initial_tau_w + [initial_tau_z])

    # b5. bounds for tax rates (remains the same)
    bounds_tau_w = [(-10.0, 10.0)] * n
    bounds_tau_z = [(1e-6, 100.0)]
    bounds = bounds_tau_w + bounds_tau_z

    # b6. minimize negative welfare using SLSQP (remains the same)
    result = minimize(swf_obj, initial_guess, method='SLSQP', bounds=bounds, constraints=[nonlinear_constraint])

    # b7. print and return results (remains the same)
    if result.success:
        opt_tau_w = result.x[0:n]
        opt_tau_z = result.x[n]
        max_welfare = -result.fun
        print("Social welfare maximization successful")
        print("Optimal tau_w:", opt_tau_w)
        print("Optimal tau_z:", opt_tau_z)
        print("Maximized Social Welfare:", max_welfare)
        return opt_tau_w, opt_tau_z, max_welfare
    else:
        print("Optimization failed")
        print("Message:", result.message)
        return None, None, None

# c. run optimization (remains the same)
xi_example_value = 0.1
optimal_tau_w, optimal_tau_z, max_welfare_value = maximize_welfare(G, xi_example_value)

# Final results printing section (remains the same, uses the fixed keys from before)
if optimal_tau_w is not None:
    print("\nResults at optimal tax rates:")
    solution, results, converged = solver.solve(optimal_tau_w, optimal_tau_z, G)
    if converged:
        print("Solution status:", results["sol_object"].status)
        print("Solution message:", results["sol_object"].message)
        print("Solution vector [T_C, T_D, log(Z_C), log(Z_D), w_c, w_d, p_D, L]:")
        print(solution)
        # ... (rest of the printing code remains the same) ...
        print("\nProduction Summary:")
        print(f"Sector C: T_prod = {results['t_c']:.4f}, z_c = {results['z_c']:.4f}")
        print(f"Sector D: T_prod = {results['t_d']:.4f}, z_d = {results['z_d']:.4f}")
        print(f"Wage vector, wage = {results['wage_vector']}, p_D = {results['p_d']:.4f}")
        print(f"Sector C output, F_C = {results['f_c']:.4f}")
        print(f"Sector D output, F_D = {results['f_d']:.4f}")

        print("\nHousehold Demands and Leisure:")
        for i in range(n):
            # Label households based on sector
            sector = 'D' if i < n_d else 'C'
            hh_idx_in_sector = i + 1 if sector == 'D' else i - n_d + 1
            print(f"Household {sector}{hh_idx_in_sector}: c = {results['c_agents'][i]:.4f}, D = {results['d_agents'][i]:.4f}, l = {results['l_agents'][i]:.4f}")

        print("\nAggregated Quantities:")
        print(f"Aggregate c = {results['agg_c']:.4f}")
        print(f"Aggregate d = {results['agg_d']:.4f}")
        print(f"Aggregate labor supply = {results['agg_labor_total_supply']:.4f}")

        print("\nLump Sum Transfer:")
        print(f"l = {results['l']:.4f}")

        print("\nFirm Profits:")
        print(f"Profit C: {results['profit_c']:.4f}")
        print(f"Profit D: {results['profit_d']:.4f}")

        print("\nHousehold Budget Constraints:")
        for i in range(n):
            sector = 'D' if i < n_d else 'C'
            hh_idx_in_sector = i + 1 if sector == 'D' else i - n_d + 1
            print(f"Household {sector}{hh_idx_in_sector}: error = {results['budget_errors'][i]:.10f}")

        market_c_residual = results['agg_c'] + 0.5 * G - results['f_c'] # Assuming G represents spending needs split 50/50 on goods?
        print(f"\nGood C market clearing residual: {market_c_residual:.10f}")
    else:
        print("Inner solver did not converge at optimal tax rates")

    if results:
         agg_polluting = results.get('z_c', 0) + results.get('z_d', 0)
         effective_utilities = results.get('utilities', np.array([]))
    else:
         agg_polluting = None
         effective_utilities = None