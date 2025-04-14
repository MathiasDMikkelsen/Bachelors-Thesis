# outer_solver.py (Baseline Modified for HeteroPoll + IC Constraints + Verification Fix)

import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
# Assuming inner_solver.py is in the same directory or path
try:
    import inner_solver as solver
    # Import parameters needed from inner_solver
    from inner_solver import n, alpha, beta, gamma, d0, t as T, phi, p_c # Ensure all needed params are imported
except ImportError:
    print("Fatal Error: Could not import 'inner_solver'.")
    print("Ensure 'inner_solver.py' exists and is accessible.")
    exit()
except AttributeError as e:
    print(f"Fatal Error: A required parameter might be missing from 'inner_solver.py': {e}")
    exit()

# --- Assert n=5 (Based on baseline phi vector) ---
assert n == 5, "This solver expects n=5 based on the imported baseline inner_solver."

# --- Parameters for Outer Solver ---
# theta: exponent for pollution damage in SWF
theta = 1.0 # Define theta here, assumed value from thesis/previous code
# G: Government consumption - will be passed as argument
# xi: Pollution aversion parameter - will be passed as argument

# --- Social Welfare Function Maximization ---

def maximize_welfare(G, xi, verify_constraints=False): # Added optional verification flag
    """
    Optimizes social welfare by choosing tau_w (vector, n=5) and tau_z (scalar),
    given G and xi, subject to IC constraints.
    Includes heterogeneous pollution effects via kappa vector.
    Optionally verifies IC constraints at the solution if verify_constraints is True.
    """

    # Define heterogeneous pollution sensitivity parameters (kappa)
    kappa = np.array([1.5, 1.25, 1.0, 0.75, 0.5]) # Example for n=5
    assert len(kappa) == n, "Length of kappa vector must match number of households (n=5)"
    sum_kappa = np.sum(kappa)

    # Define the objective function (negative social welfare)
    def swf_obj(x):
        # --- (swf_obj function remains the same as in the previous version) ---
        tau_w = x[0:n]
        tau_z = x[n]

        if tau_z <= 0: return 1e12

        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged or results is None: return 1e10
            utilities = results.get('utilities', None)
            z_c = results.get('z_c', None)
            z_d = results.get('z_d', None)
            if utilities is None or z_c is None or z_d is None: return 1e11
            if np.any(np.isinf(utilities)) or np.any(np.isnan(utilities)): return 1e9
            agg_polluting = z_c + z_d
            if agg_polluting < 0: return 1e8
            blue_welfare_sum = np.sum(utilities)
            green_disutility_total = sum_kappa * xi * (agg_polluting ** theta)
            welfare = blue_welfare_sum - green_disutility_total
            return -welfare
        except Exception as e:
            print(f"Error occurred in swf_obj for x={x}: {e}")
            import traceback
            traceback.print_exc()
            return 1e12
        # --- (End of swf_obj) ---

    # Define Incentive Compatibility constraints function (nested is fine here)
    def ic_constraints(x):
        # --- (ic_constraints function remains the same as in the previous version) ---
        tau_w = x[0:n]
        tau_z = x[n]
        if tau_z <= 0: return -np.ones(n * (n - 1)) * 1e6
        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged or results is None: return -np.ones(n * (n - 1)) * 1e6
            l_agents = results.get('l_agents')
            w = results.get('w')
            c_agents = results.get('c_agents')
            d_agents = results.get('d_agents')
            current_utilities = results.get('utilities')
            if l_agents is None or w is None or c_agents is None or d_agents is None or current_utilities is None:
                 return -np.ones(n * (n - 1)) * 1e6
            if w <= 0: return -np.ones(n * (n - 1)) * 1e6
            if np.any(l_agents > T) or np.any(l_agents < 0): return -np.ones(n * (n - 1)) * 1e6
            pre_tax_income_j = phi * w * (T - l_agents)
            g_list = []
            for i in range(n):
                U_i = current_utilities[i]
                if U_i <= -1e8:
                     for j in range(n):
                         if i != j: g_list.append(1e6)
                     continue
                for j in range(n):
                    if i == j: continue
                    c_j = c_agents[j]; d_j = d_agents[j]
                    if phi[i] * w == 0: ell_i_j = -1
                    else:
                        labor_i_needed = pre_tax_income_j[j] / (phi[i] * w)
                        ell_i_j = T - labor_i_needed
                    if ell_i_j <= 0 or ell_i_j >= T or c_j <= 0 or d_j <= d0: U_i_j = -np.inf
                    else:
                        try: U_i_j = (alpha * np.log(c_j) + beta * np.log(d_j - d0) + gamma * np.log(ell_i_j))
                        except ValueError: U_i_j = -np.inf
                    if np.isinf(U_i_j) and U_i_j < 0: g_list.append(1e9)
                    else: g_list.append(U_i - U_i_j)
            return np.array(g_list)
        except Exception as e:
            print(f"Error evaluating IC constraints for x={x}: {e}")
            import traceback
            traceback.print_exc()
            return -np.ones(n * (n - 1)) * 1e6
        # --- (End of ic_constraints) ---

    # Create the NonlinearConstraint object
    nonlinear_constraint = NonlinearConstraint(ic_constraints, lb=0, ub=np.inf)

    # Initial guess
    initial_tau_w = np.zeros(n)
    initial_tau_z = 1.0
    initial_guess = np.concatenate([initial_tau_w, [initial_tau_z]])

    # Bounds
    tau_w_bounds = [(-10.0, 10.0)] * n
    tau_z_bounds = [(1e-6, 100.0)]
    bounds = tau_w_bounds + tau_z_bounds

    # Perform optimization
    optimal_tau_w = None
    optimal_tau_z = None
    max_welfare_value = None
    try:
        result = minimize(
            swf_obj,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=[nonlinear_constraint], # Include IC constraints
            options={'disp': False, 'ftol': 1e-7, 'maxiter': 500}
        )

        if result.success:
            optimal_tau_w = result.x[0:n]
            optimal_tau_z = result.x[n]
            max_welfare_value = -result.fun

            # --- MOVED VERIFICATION LOGIC HERE ---
            if verify_constraints:
                print(f"\n--- Verifying results for G={G}, xi={xi} ---")
                try:
                    solution_final, results_final, converged_final = solver.solve(optimal_tau_w, optimal_tau_z, G)
                    if converged_final and results_final is not None:
                        print("  Inner solver converged successfully at the optimum.")
                        # Verify IC constraints at the solution using the *local* ic_constraints function
                        ic_values_final = ic_constraints(result.x) # Use result.x directly
                        if np.all(ic_values_final >= -1e-6): # Allow small tolerance
                             print("  IC constraints appear satisfied at the optimum.")
                        else:
                             print("  WARNING: IC constraints appear VIOLATED at the optimum.")
                             # print(f"  IC values: {ic_values_final}") # Optional detail
                    else:
                        print("  Warning: Inner solver FAILED to converge at the found optimum during verification.")
                except Exception as e_verify:
                   print(f"  Error during verification step: {e_verify}")
                print("--- End Verification ---")
            # --- END OF MOVED VERIFICATION ---

        else:
            print(f"Optimization failed for G={G}, xi={xi} (IC Constraints ON, HeteroPoll)")
            print(f"Message: {result.message}")
            # Return None if failed
            optimal_tau_w = None
            optimal_tau_z = None
            max_welfare_value = None

    except Exception as e:
        print(f"Fatal error during optimization call for G={G}, xi={xi}: {e}")
        import traceback
        traceback.print_exc()
        # Ensure None is returned on fatal error
        optimal_tau_w = None
        optimal_tau_z = None
        max_welfare_value = None

    return optimal_tau_w, optimal_tau_z, max_welfare_value # Return results or None


# --- Example Usage (Optional for testing this file directly) ---
if __name__ == "__main__":
    print("--- Running direct test of outer_solver.py (Baseline + HeteroPoll + IC Constraints + Fix) ---")
    G_test = 5.0
    xi_test = 0.1 # Example value

    print(f"Attempting optimization for G={G_test}, xi={xi_test} with verification...")
    # Call with verify_constraints=True
    opt_w, opt_z, max_sw = maximize_welfare(G_test, xi_test, verify_constraints=True)

    if opt_w is not None:
        print("\nOptimization Test Completed Successfully (details in verification block above):")
        print(f"  Optimal tau_w: {opt_w}")
        print(f"  Optimal tau_z: {opt_z}")
        print(f"  Maximized SWF: {max_sw}")
    else:
        print("\nOptimization Test Failed.")
    print("--- End of direct test ---")