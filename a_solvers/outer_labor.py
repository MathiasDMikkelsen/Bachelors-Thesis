import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
# Ensure the inner solver module name matches your file name (e.g., inner_labor.py)
import inner_labor as solver
# Import necessary parameters from the inner solver
from inner_labor import alpha, beta, gamma, d0, phi, n # Removed t import as T is defined below

# a. parameters
T = 24      # Assuming T=24 is the correct time endowment
theta = 1.0
G = 1.0

# b. maximize social welfare
def maximize_welfare(G, xi):

    # b1. define objective function
    def swf_obj(x):

        tau_w = x[0:n]
        tau_z = x[n]

        try:
            # Call the inner solver
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return 1e10 # penalize inner layer non-convergence

            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d'] # aggregate pollution from inner solver

            welfare = np.sum(utilities) - 5 * xi * (agg_polluting ** theta) # Assuming the factor 5 is intended
            return -welfare # minimize negative welfare

        except Exception as e:
            print(f"Solver failed with error: {e}")
            return 1e10 # penalize inner layer failure

    # b2. define IC constraints
    def ic_constraints(x):

        tau_w = x[0:n]
        tau_z = x[n]

        try:
            # Call the inner solver again for constraint evaluation
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                # Return large negative values if inner solver fails, indicating constraint violation
                return -np.ones(n * (n - 1)) * 1e6 # penalize if inner solver fails

            # Pre-calculate terms needed for IC checks based on the provided logic
            I = np.zeros(n)
            wage_vector = results['wage_vector'] # Use the wage vector from results
            l_agents = results['l_agents'] # Use leisure vector from results
            for j in range(n):
                # Use wage_vector (instead of results['w']) to compute income for household j
                # The original formula was: I[j] = (T - results['l_agents'][j]) * (1.0 - tau_w[j]) * phi[j] * results['wage_vector'][j]
                # Let's stick to the provided calculation
                labor_supply_j = T - l_agents[j]
                I[j] = labor_supply_j * (1.0 - tau_w[j]) * phi[j] * wage_vector[j] # Original income calc used

            g_list = []
            utilities = results['utilities'] # Get actual utilities
            c_agents = results['c_agents'] # Get actual consumption c
            d_agents = results['d_agents'] # Get actual consumption d
            for i in range(n):
                U_i = utilities[i]
                for j in range(n):
                    if i == j:
                        continue

                    c_j = c_agents[j] # Consumption bundle of agent j
                    d_j = d_agents[j] # Consumption bundle of agent j

                    # Use wage_vector for household i when computing the denominator
                    # Denominator for agent i considering mimicking j under j's tax
                    denom = (1.0 - tau_w[j]) * phi[i] * wage_vector[i] # Original denom calc used
                    if denom <= 1e-9: # Avoid division by zero (handle potential non-positive wages/net rates)
                        # If agent i cannot earn positive income with j's tax rate, they cannot mimic this way.
                        # Constraint is satisfied (U_i - U_i_j -> U_i - (-inf) -> +inf)
                        g_list.append(1e6) # Assign large positive value
                        continue

                    # Compute leisure for household i pretending to be j (using original formula)
                    ell_i_j = T - I[j] / denom # Original leisure calc used

                    U_i_j = -np.inf # Default to -inf if mimicking choice is invalid
                    # Check validity of mimicking choices before calculating utility
                    if ell_i_j > 1e-9 and ell_i_j < T - 1e-9 and c_j > 1e-9 and d_j > d0 + 1e-9:
                         # Calculate utility agent i gets from mimicking agent j
                         try:
                             # Use logs carefully
                            log_c_j = np.log(c_j)
                            log_d_j_net = np.log(d_j - d0)
                            log_ell_i_j = np.log(ell_i_j)
                            U_i_j = (alpha * log_c_j + beta * log_d_j_net + gamma * log_ell_i_j)
                         except ValueError: # Catch log(<=0) just in case
                             U_i_j = -np.inf # Assign -inf if log fails

                    # Constraint: U_i >= U_i_j --> U_i - U_i_j >= 0
                    constraint_value = U_i - U_i_j
                    # Replace inf values resulting from U_i - (-inf) before appending
                    if np.isinf(constraint_value):
                       g_list.append(1e6) # Treat U_i - (-inf) as satisfied
                    else:
                       g_list.append(constraint_value)


            return np.array(g_list)

        except Exception as e:
            print(f"ic constraint evaluation failed with error: {e}")
            # Return large negative values to indicate constraint violation
            return -np.ones(n * (n - 1)) * 1e6 # penalize if evaluation fails

    # b3. set up the nonlinear constraint for the IC conditions
    # We need n * (n-1) constraints, all >= 0
    num_constraints = n * (n - 1)
    nonlinear_constraint = NonlinearConstraint(ic_constraints, lb=np.zeros(num_constraints), ub=np.inf)

    # b4. initial guess
    initial_tau_w = [0.0] * n
    initial_tau_z = 0.5 # Original guess
    initial_guess = np.array(initial_tau_w + [initial_tau_z])

    # b5. bounds for tax rates
    # Using the original bounds provided
    bounds_tau_w = [(-10.0, 10.0)] * n
    bounds_tau_z = [(1e-6, 100.0)]
    bounds = bounds_tau_w + bounds_tau_z

    # b6. minimize negative welfare using SLSQP
    # Using default options as in the original code
    result = minimize(swf_obj, initial_guess, method='SLSQP', bounds=bounds, constraints=[nonlinear_constraint])

    # b7. print and return results
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

# c. run optimization
xi_example_value = 0.1 # example value for xi
optimal_tau_w, optimal_tau_z, max_welfare_value = maximize_welfare(G, xi_example_value) # Renamed max_welfare -> max_welfare_value

# If optimization was successful, solve the inner model one last time with optimal taxes
# to display the full equilibrium details.
if optimal_tau_w is not None:
    print("\nResults at optimal tax rates:")
    # Ensure G is passed correctly if it was intended to be used from the outer scope
    solution, results, converged = solver.solve(optimal_tau_w, optimal_tau_z, G)

    if converged:
        # --- FIXED KEY ACCESS HERE ---
        print("Solution status:", results["sol_object"].status) # Changed 'sol' to 'sol_object'
        print("Solution message:", results["sol_object"].message) # Changed 'sol' to 'sol_object'
        print("Solution vector [T_C, T_D, log(Z_C), log(Z_D), w_c, w_d, p_D, L]:")
        print(solution)

        print("\nProduction Summary:")
        print(f"Sector C: T_prod = {results['t_c']:.4f}, z_c = {results['z_c']:.4f}")
        print(f"Sector D: T_prod = {results['t_d']:.4f}, z_d = {results['z_d']:.4f}")
        # Original print statement kept:
        print(f"Wage vector, wage = {results['wage_vector']}, p_D = {results['p_d']:.4f}")
        print(f"Sector C output, F_C = {results['f_c']:.4f}")
        print(f"Sector D output, F_D = {results['f_d']:.4f}")

        print("\nHousehold Demands and Leisure:")
        for i in range(n):
            print(f"Household {i+1}: c = {results['c_agents'][i]:.4f}, D = {results['d_agents'][i]:.4f}, l = {results['l_agents'][i]:.4f}")

        print("\nAggregated Quantities:")
        print(f"Aggregate c = {results['agg_c']:.4f}")
        print(f"Aggregate d = {results['agg_d']:.4f}")
        # --- FIXED KEY ACCESS HERE ---
        print(f"Aggregate labor supply = {results['agg_labor_total_supply']:.4f}") # Changed 'agg_labor' to 'agg_labor_total_supply'

        print("\nLump Sum Transfer:")
        print(f"l = {results['l']:.4f}")

        print("\nFirm Profits:")
        print(f"Profit C: {results['profit_c']:.4f}")
        print(f"Profit D: {results['profit_d']:.4f}")

        print("\nHousehold Budget Constraints:")
        for i in range(n):
            print(f"Household {i+1}: error = {results['budget_errors'][i]:.10f}") # Kept original precision

        # Kept original market clearing check (Note: uses 0.5*G, ensure this matches inner solver if G is used there for goods)
        market_c_residual = results['agg_c'] + 0.5 * G - results['f_c']
        print(f"\nGood C market clearing residual: {market_c_residual:.10f}") # Kept original precision
    else:
        print("Inner solver did not converge at optimal tax rates")

    # These lines were outside the `if converged:` block in the original, kept that way
    # Note: results might not be fully populated if converged is False
    if results: # Basic check if results dictionary exists
         agg_polluting = results.get('z_c', 0) + results.get('z_d', 0) # Use .get for safety if not converged
         effective_utilities = results.get('utilities', np.array([])) # Use .get for safety
    else:
         agg_polluting = None
         effective_utilities = None


# Note: The lines below were calculating utilities/pollution again outside the function
# They are kept here but might rely on 'results' being valid from the last call within the 'if' block.
# Consider moving them inside the 'if converged:' block if they should only run on success.
# if optimal_tau_w is not None and converged: # Check convergence again?
#      agg_polluting = results['z_c'] + results['z_d']
#      effective_utilities = results['utilities']