import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import warnings

# --- Ensure the inner solver module name matches your file name ---
# Assuming the latest 2-household solver is named 'inner_solver.py'
try:
    import inner_solver_ext as solver
    # Import necessary parameters from the NEW inner solver
    from inner_solver_ext import alpha, beta, gamma, d0, t as T, n, n_d, n_c # n=2, n_d=1, n_c=1
except ImportError:
    print("Error: Could not import 'inner_solver'.")
    print("Please ensure the modified 2-household inner solver file is named 'inner_solver.py' and is accessible.")
    exit()
# ---------------------------------------------------------------

# a. parameters
# T is now imported from inner_solver
theta = 1.0 # Pollution parameter in SWF
G = 5.0     # Government spending requirement

# --- Assertions based on expected inner solver ---
assert n == 2, "This outer solver requires the inner solver to have n=2 households."
assert n_d == 1, "This outer solver requires the inner solver to have n_d=1."
assert n_c == 1, "This outer solver requires the inner solver to have n_c=1."
# -------------------------------------------------


# b. maximize social welfare function
def maximize_welfare(target_G, xi):
    """
    Finds optimal taxes (tau_w_d, tau_w_c, tau_z) for the 2-household model.

    Args:
        target_G (float): The required government spending level.
        xi (float): The weight on pollution disutility in the SWF.

    Returns:
        tuple: (optimal_tau_w, optimal_tau_z, max_welfare) or (None, None, None) if fails.
    """
    print(f"\n--- Starting Welfare Maximization (2 Households) for G={target_G}, xi={xi} ---")

    # b1. define objective function (negative social welfare)
    def swf_obj(x):
        # --- MODIFIED: Extract 2 tau_w rates and tau_z ---
        tau_w = x[0:n] # x[0] is tau_w_d, x[1] is tau_w_c
        tau_z = x[n]   # x[2] is tau_z
        # -------------------------------------------------

        try:
            solution, results, converged = solver.solve(tau_w, tau_z, target_G)
            if not converged:
                return 1e10 # Penalize non-convergence

            utilities = results['utilities'] # Should be length 2 [u_d, u_c]
            # Check for invalid utility values
            if np.any(np.isinf(utilities)) or np.any(np.isnan(utilities)):
                 print(f"Warning: Invalid utilities found: {utilities}")
                 return 1e10

            agg_polluting = results['z_c'] + results['z_d']

            # Calculate SWF (ensure consistency with thesis/model, e.g., log utility)
            # Using sum(util) - penalty form provided previously
            welfare = np.sum(utilities) - 2*xi * (agg_polluting**theta) # Factor 5 was in prev code
            return -welfare

        except Exception as e:
            print(f"!!! Inner solver failed during SWF evaluation: {e} for tau_w={tau_w}, tau_z={tau_z}")
            return 1e10

    # b2. define the two relevant IC constraints
    def ic_constraints(x):
        # --- MODIFIED: Extract 2 tau_w rates and tau_z ---
        tau_w = x[0:n] # x[0] is tau_w_d, x[1] is tau_w_c
        tau_z = x[n]   # x[2] is tau_z
        # -------------------------------------------------

        try:
            solution, results, converged = solver.solve(tau_w, tau_z, target_G)
            # --- MODIFIED: Return penalty array of size 2 ---
            if not converged:
                return np.array([-1e6, -1e6])
            # -----------------------------------------------

            # Extract necessary results (arrays of size 2)
            utilities = results['utilities']           # [u_d, u_c]
            c_agents = results['c_agents']             # [c_d, c_c]
            d_agents = results['d_agents']             # [d_d, d_c]
            l_agents = results['l_agents']             # [l_d, l_c]
            wage_vector = results['wage_vector']       # [w_d, w_c]
            labor_supply_agents = results['labor_supply_agents'] # [lab_d, lab_c]

            # Check for invalid utilities
            if np.any(np.isinf(utilities)) or np.any(np.isnan(utilities)):
                 print(f"Warning: Invalid utilities in IC check: {utilities}")
                 return np.array([-1e6, -1e6]) # Penalize

            g_list = [] # Should contain 2 elements: [g0, g1]

            # --- Constraint 1: Agent D (i=0) mimicking Agent C (j=1) ---
            i = 0 # Dirty worker
            j = 1 # Clean worker
            U_i = utilities[i]

            # Income agent C actually earned (needed for agent D to mimic)
            # --- MODIFIED: No phi, use labor_supply_agents ---
            I_j = labor_supply_agents[j] * (1.0 - tau_w[j]) * wage_vector[j]
            # Denominator for agent i's earnings if they face agent j's tax
            denom = (1.0 - tau_w[j]) * wage_vector[i] # w_d * (1-tau_w_c)
            # --------------------------------------------------

            ell_i_j = -np.inf # Default leisure if cannot mimic
            if denom > 1e-9: # Check if agent i can earn income under j's tax rate
                ell_i_j = T - I_j / denom

            U_i_j = -np.inf # Default utility if cannot mimic
            # Check validity of mimicking choices (c_j, d_j from actual j; ell_i_j for i)
            if ell_i_j > 1e-9 and ell_i_j < T - 1e-9 and c_agents[j] > 1e-9 and d_agents[j] > d0 + 1e-9:
                try:
                    log_c_j = np.log(c_agents[j])
                    log_d_j_net = np.log(d_agents[j] - d0)
                    log_ell_i_j = np.log(ell_i_j)
                    # Calculate utility i gets from mimicking j's bundle with own leisure ell_i_j
                    U_i_j = (alpha * log_c_j + beta * log_d_j_net + gamma * log_ell_i_j)
                    # ADD POLLUTION TERM IF NEEDED (using agent i's sensitivity if heterogeneous)
                    # e.g., U_i_j += e0 - pollution_disutility_i
                except ValueError:
                    U_i_j = -np.inf

            constraint_value_0 = U_i - U_i_j
            g_list.append(constraint_value_0 if not np.isinf(constraint_value_0) else 1e6)

            # --- Constraint 2: Agent C (i=1) mimicking Agent D (j=0) ---
            i = 1 # Clean worker
            j = 0 # Dirty worker
            U_i = utilities[i]

            # Income agent D actually earned
            # --- MODIFIED: No phi, use labor_supply_agents ---
            I_j = labor_supply_agents[j] * (1.0 - tau_w[j]) * wage_vector[j]
            # Denominator for agent i's earnings if they face agent j's tax
            denom = (1.0 - tau_w[j]) * wage_vector[i] # w_c * (1-tau_w_d)
            # --------------------------------------------------

            ell_i_j = -np.inf
            if denom > 1e-9:
                ell_i_j = T - I_j / denom

            U_i_j = -np.inf
            if ell_i_j > 1e-9 and ell_i_j < T - 1e-9 and c_agents[j] > 1e-9 and d_agents[j] > d0 + 1e-9:
                try:
                    log_c_j = np.log(c_agents[j])
                    log_d_j_net = np.log(d_agents[j] - d0)
                    log_ell_i_j = np.log(ell_i_j)
                    U_i_j = (alpha * log_c_j + beta * log_d_j_net + gamma * log_ell_i_j)
                    # ADD POLLUTION TERM IF NEEDED (using agent i's sensitivity)
                except ValueError:
                    U_i_j = -np.inf

            constraint_value_1 = U_i - U_i_j
            g_list.append(constraint_value_1 if not np.isinf(constraint_value_1) else 1e6)

            return np.array(g_list) # Should be size 2

        except Exception as e:
            print(f"!!! IC constraint evaluation failed: {e}")
            # --- MODIFIED: Return penalty array of size 2 ---
            return np.array([-1e6, -1e6])
            # -----------------------------------------------

    # b3. define nonlinear constraint object (for 2 constraints)
    # --- MODIFIED: Only 2 constraints ---
    num_constraints = 2
    nonlinear_constraint = NonlinearConstraint(ic_constraints, lb=np.zeros(num_constraints), ub=np.inf)
    # -----------------------------------

    # b4. initial guess [tau_w_d, tau_w_c, tau_z]
    # --- MODIFIED: Length 3 initial guess ---
    initial_tau_w = [0.1, 0.1] # Guess for [tau_w_d, tau_w_c]
    initial_tau_z = 1.0
    initial_guess = np.array(initial_tau_w + [initial_tau_z])
    # ---------------------------------------

    # b5. bounds for tax rates [bound_d, bound_c, bound_z]
    # --- MODIFIED: Length 3 bounds ---
    bounds_tau_w = [(-5.0, 5.0)] * n # Bounds for tau_w_d, tau_w_c
    bounds_tau_z = [(1e-4, 50.0)]    # Bounds for tau_z
    bounds = bounds_tau_w + bounds_tau_z
    # ----------------------------------

    # b6. minimize negative welfare using SLSQP
    options = {'disp': True, 'ftol': 1e-7, 'maxiter': 200}
    result = minimize(swf_obj,
                     initial_guess,
                     method='SLSQP',
                     bounds=bounds,
                     constraints=[nonlinear_constraint],
                     options=options)

    # b7. print and return results
    print("--- Welfare Maximization Finished ---")
    if result.success:
        # --- MODIFIED: Extract 2 tau_w and 1 tau_z ---
        opt_tau_w = result.x[0:n] # [opt_tau_w_d, opt_tau_w_c]
        opt_tau_z = result.x[n]
        # --------------------------------------------
        max_welfare = -result.fun
        print("\nSocial welfare maximization successful")
        print(f"Optimal tau_w: [tau_w_d={opt_tau_w[0]:.6f}, tau_w_c={opt_tau_w[1]:.6f}]")
        print(f"Optimal tau_z: {opt_tau_z:.6f}")
        print(f"Maximized Social Welfare: {max_welfare:.6f}")
        # Check final constraints
        try:
             final_constraints = ic_constraints(result.x)
             print(f"IC Constraints at optimum: [D>=D(C): {final_constraints[0]:.4e}, C>=C(D): {final_constraints[1]:.4e}] (should be >= 0)")
        except:
             print("Could not evaluate constraints at final point.")
        return opt_tau_w, opt_tau_z, max_welfare
    else:
        print("\nOptimization failed")
        print("Message:", result.message)
        print("Final Objective Function Value:", result.fun)
        print("Final Solution Vector (x):", result.x)
        try:
             final_constraints = ic_constraints(result.x)
             print(f"IC Constraints at final point: [D>=D(C): {final_constraints[0]:.4e}, C>=C(D): {final_constraints[1]:.4e}] (should be >= 0)")
        except:
             print("Could not evaluate constraints at final point.")

        return None, None, None

# --- c. Run Optimization ---
xi_example_value = 0.1 # Pollution aversion parameter
optimal_tau_w, optimal_tau_z, max_welfare_value = maximize_welfare(G, xi_example_value)

# --- d. Print results at optimal taxes ---
if optimal_tau_w is not None and optimal_tau_z is not None:
    print("\n--- Results at Optimal Taxes ---")
    try:
        solution, results, converged = solver.solve(optimal_tau_w, optimal_tau_z, G)

        if converged and results:
            print("Inner solver converged successfully with optimal taxes.")
            print("Solution status:", results.get("sol_object", {}).status) # Safer access
            print("Solution message:", results.get("sol_object", {}).message)

            print("\nOptimal Equilibrium Details:")
            print(f"  Solution vector [t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l]:\n  {solution}")

            print("\n  Prices & Transfers:")
            print(f"    p_c = {solver.p_c:.4f} (Numeraire)")
            print(f"    p_d = {results['p_d']:.4f}")
            print(f"    w_c = {results['w_c']:.4f}")
            print(f"    w_d = {results['w_d']:.4f}")
            print(f"    Lump-sum transfer (l) = {results['l']:.4f}")

            print("\n  Production & Inputs:")
            print(f"    Sector C: Labor Demand (t_c) = {results['t_c']:.4f}, Z Input (z_c) = {results['z_c']:.4f}, Output (f_c) = {results['f_c']:.4f}")
            print(f"    Sector D: Labor Demand (t_d) = {results['t_d']:.4f}, Z Input (z_d) = {results['z_d']:.4f}, Output (f_d) = {results['f_d']:.4f}")
            print(f"    Total Polluting Input (z_c + z_d) = {results['z_c'] + results['z_d']:.4f}")

            print("\n  Labor Market:")
            print(f"    Clean Sector: Supply/Demand = {results['agg_labor_c']:.4f}") # Labor supply = demand in equilibrium
            print(f"    Dirty Sector: Supply/Demand = {results['agg_labor_d']:.4f}")
            print(f"    Total Labor = {results['agg_labor_total']:.4f}")

            print("\n  Household Details (Agent D: index 0, Agent C: index 1):")
            print(f"    Agent D (Dirty): tau_w={optimal_tau_w[0]:.4f}, c={results['c_agents'][0]:.4f}, d={results['d_agents'][0]:.4f}, leisure={results['l_agents'][0]:.4f}, labor={results['labor_supply_agents'][0]:.4f}, utility={results['utilities'][0]:.4f}")
            print(f"    Agent C (Clean): tau_w={optimal_tau_w[1]:.4f}, c={results['c_agents'][1]:.4f}, d={results['d_agents'][1]:.4f}, leisure={results['l_agents'][1]:.4f}, labor={results['labor_supply_agents'][1]:.4f}, utility={results['utilities'][1]:.4f}")

            print("\n  Aggregate Consumption:")
            print(f"    Aggregate c = {results['agg_c']:.4f}")
            print(f"    Aggregate d = {results['agg_d']:.4f}")

            print("\n  Checks:")
            print(f"    Firm C Profit = {results['profit_c']:.4e}")
            print(f"    Firm D Profit = {results['profit_d']:.4e}")
            print(f"    Budget Error (Agent D) = {results['budget_errors'][0]:.4e}")
            print(f"    Budget Error (Agent C) = {results['budget_errors'][1]:.4e}")
            print("\n  System Residuals at Optimum:")
            for i, res in enumerate(results['system_residuals']):
                print(f"    Eq {i+1}: {res:.4e}")
        else:
             print("!!! Inner solver FAILED to converge with the optimal taxes found.")

    except Exception as e:
        print(f"\n!!! An error occurred when running the inner solver with optimal taxes: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\nOuter optimization failed, cannot show equilibrium details for optimal taxes.")