import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import a_inner_solver as solver
from a_inner_solver import alpha, beta, gamma, d0, phi, n

# a. parameters
T = 24
xi = 1.0 # Removed this line
theta = 1.0
G = 5.0

# b. maximize social welfare
def maximize_welfare(G, xi): # <-- Changed this line only

    # b1. define objective function
    def swf_obj(x):

        tau_w = x[0:n]
        tau_z = x[n]

        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return 1e10  # penalize inner layer non-convergence

            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d'] # extract inner layer solution

            welfare = np.sum(utilities)-5*xi * (agg_polluting**theta)
            return -welfare  # calculate and return negative welfare

        except Exception as e:
            print(f"Solver failed with error: {e}")
            return 1e10  # penalize inner layer failiure

    # b2. define ic constraints
    def ic_constraints(x):

        tau_w = x[0:n]
        tau_z = x[n]

        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return -np.ones(n*(n-1)) * 1e6  # penalize inner layer non-convergence

            I = np.zeros(n)
            for j in range(n):
                I[j] = (T - results['l_agents'][j])*(1.0 - tau_w[j])*phi[j]*results['w'] # compute income for hh j

            g_list = []

            for i in range(n):
                U_i = results['utilities'][i]
                for j in range(n):
                    if i == j:
                        continue

                    c_j = results['c_agents'][j]
                    d_j = results['d_agents'][j] # retrieve consumption for hh j

                    denom = (1.0 - tau_w[j]) * phi[i] * results['w']
                    if denom == 0: # ial: changed from "<=0" to only avoid division by zero
                        g_list.append(-1e6)
                        continue

                    ell_i_j = T - I[j] / denom # comute leisure for hh i pretending to be hh j

                    if ell_i_j <= 0:
                        U_i_j = -1e6 # throw away corner solutions
                    else:
                        if c_j <= 0 or d_j <= d0:
                            U_i_j = -1e6 # throw away corner solutions
                        else:
                            U_i_j = (alpha * np.log(c_j) +
                                     beta * np.log(d_j - d0) +
                                     gamma * np.log(ell_i_j)) # calculate utility for hh i pretending to be hh j

                    g_list.append(U_i - U_i_j) # return two ic constraints for each pair of hh

            return np.array(g_list)

        except Exception as e:
            print(f"ic constraint evaluation failed with error: {e}")
            return -np.ones(n*(n-1)) * 1e6  # penzlize if ic constraints cannot be evaluated

    # b3. define ic constraints as a nonlinear constraint
    nonlinear_constraint = NonlinearConstraint(ic_constraints, lb=0, ub=np.inf)

    # b4. initial guess.
    initial_tau_w = n*[0.1]# actually guess does not matter much, model converges to same solution. choose initial_tau_w = [-2.5, -0.5, -0.2, 0.1, 0.5] if close to klenert
    initial_tau_z = 0.1
    initial_guess = np.array(initial_tau_w + [initial_tau_z])

    # b5. bounds for tax rates
    bounds = [(-1.0, 1.0)] * n + [(1e-6, 100.0)]

    # b6. minimize negative welfare using slsqp
    result = minimize(swf_obj, initial_guess, method='SLSQP', bounds=bounds, constraints=[nonlinear_constraint])

    # b7. print results
    if result.success:
        opt_tau_w = result.x[0:n]
        opt_tau_z = result.x[n]
        max_welfare = -result.fun

        print("social welfare maximization successful")
        print("optimal tau_w:", opt_tau_w)
        print("optimal tau_z:", opt_tau_z)
        print("maximized Social Welfare:", max_welfare)
        return opt_tau_w, opt_tau_z, max_welfare
    else:
        print("optimization failed")
        print("message:", result.message)
        return None, None, None

# c. run optimization
xi_example_value = 0.1 # Or any other example value you want to test
optimal_tau_w, optimal_tau_z, max_welfare = maximize_welfare(G, xi_example_value)

if optimal_tau_w is not None:
    print("\nresults at optimal tax rates:")
    solution, results, converged = solver.solve(optimal_tau_w, optimal_tau_z, G)

    if converged:
        print("solution status:", results["sol"].status)
        print("solution message:", results["sol"].message)
        print("solution vector [T_C, T_D, Z_C, Z_D, w, p_D, L]:")
        print(solution)

        print("\nproduction Summary:")
        print(f"sector C: T_prod = {results['t_c']:.4f}, z_c = {results['z_c']:.4f}")
        print(f"sector D: T_prod = {results['t_d']:.4f}, z_d = {results['z_d']:.4f}")
        print(f"common wage, w = {results['w']:.4f}, p_D = {results['p_d']:.4f}")
        print(f"sector C output, F_C = {results['f_c']:.4f}")
        print(f"sector D output, F_D = {results['f_c']:.4f}")

        print("\nhousehold Demands and Leisure:")
        for i in range(n):
            print(f"household {i+1}: c = {results['c_agents'][i]:.4f}, D = {results['d_agents'][i]:.4f}, l = {results['l_agents'][i]:.4f}")

        print("\naggregated Quantities:")
        print(f"aggregate c = {results['agg_c']:.4f}")
        print(f"aggregate d = {results['agg_d']:.4f}")
        print(f"aggregate labor supply = {results['agg_labor']:.4f}")

        print("\nlump sum:")
        print(f"l = {results['l']:.4f}")

        print("\nfirm profits:")
        print(f"profit c: {results['profit_c']:.4f}")
        print(f"profit d: {results['profit_d']:.4f}")

        print("\nhousehold budget constraints:")
        for i in range(n):
            print(f"household {i+1}: error = {results['budget_errors'][i]:.10f}")

        print(f"\ngood c market clearing residual: {(results['agg_c'] + 0.5*G) - results['f_c']:.10f}")
    else:
        print("inner solver did not converge at optimal tax rates")

    agg_polluting = results['z_c'] + results['z_d']
effective_utilities = results['utilities']

# Define a function to compute the Gini coefficient
def gini(array):
    # Convert to numpy array and flatten
    array = np.array(array).flatten()
    # Sort the array
    sorted_array = np.sort(array)
    n = array.size
    # Create index numbers (1-indexed)
    index = np.arange(1, n+1)
    # Compute Gini coefficient using the standard formula
    return (np.sum((2 * index - n - 1) * sorted_array)) / (n * np.sum(sorted_array))

# Calculate the Gini coefficient for effective utilities
gini_value = gini(effective_utilities)
print("Gini coefficient in effective utility:", gini_value)