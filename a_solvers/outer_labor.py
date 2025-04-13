import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import a_solvers.inner_labor as solver
from a_solvers.inner_labor import alpha, beta, gamma, d0, phi, n

# a. parameters
T = 24
theta = 1.0
G = 1.0

# b. maximize social welfare
def maximize_welfare(G, xi):

    # b1. define objective function
    def swf_obj(x):

        tau_w = x[0:n]
        tau_z = x[n]

        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return 1e10  # penalize inner layer non-convergence

            utilities = results['utilities']
            agg_polluting = results['z_c'] + results['z_d']  # aggregate pollution from inner solver

            welfare = np.sum(utilities) - 5 * xi * (agg_polluting ** theta)
            return -welfare  # minimize negative welfare

        except Exception as e:
            print(f"Solver failed with error: {e}")
            return 1e10  # penalize inner layer failure

    # b2. define IC constraints
    def ic_constraints(x):

        tau_w = x[0:n]
        tau_z = x[n]

        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return -np.ones(n * (n - 1)) * 1e6  # penalize if inner solver fails

            I = np.zeros(n)
            for j in range(n):
                # Use wage_vector (instead of results['w']) to compute income for household j
                I[j] = (T - results['l_agents'][j]) * (1.0 - tau_w[j]) * phi[j] * results['wage_vector'][j]

            g_list = []
            for i in range(n):
                U_i = results['utilities'][i]
                for j in range(n):
                    if i == j:
                        continue

                    c_j = results['c_agents'][j]
                    d_j = results['d_agents'][j]  # consumption for household j

                    # Use wage_vector for household i when computing the denominator
                    denom = (1.0 - tau_w[j]) * phi[i] * results['wage_vector'][i]
                    if denom == 0:
                        g_list.append(-1e6)
                        continue

                    ell_i_j = T - I[j] / denom  # compute leisure for household i pretending to be j

                    if ell_i_j <= 0:
                        U_i_j = -1e6  # penalize corner solutions
                    else:
                        if c_j <= 0 or d_j <= d0:
                            U_i_j = -1e6  # penalize corner solutions
                        else:
                            U_i_j = (alpha * np.log(c_j) +
                                     beta * np.log(d_j - d0) +
                                     gamma * np.log(ell_i_j))
                    g_list.append(U_i - U_i_j)

            return np.array(g_list)

        except Exception as e:
            print(f"ic constraint evaluation failed with error: {e}")
            return -np.ones(n * (n - 1)) * 1e6  # penalize if evaluation fails

    # b3. set up the nonlinear constraint for the IC conditions
    nonlinear_constraint = NonlinearConstraint(ic_constraints, lb=0, ub=np.inf)

    # b4. initial guess
    initial_tau_w = [0.0] * n  
    initial_tau_z = 0.5
    initial_guess = np.array(initial_tau_w + [initial_tau_z])

    # b5. bounds for tax rates
    bounds = [(-10.0, 10.0)] * n + [(1e-6, 100.0)]

    # b6. minimize negative welfare using SLSQP
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
xi_example_value = 0.1  # example value for xi
optimal_tau_w, optimal_tau_z, max_welfare = maximize_welfare(G, xi_example_value)

if optimal_tau_w is not None:
    print("\nResults at optimal tax rates:")
    solution, results, converged = solver.solve(optimal_tau_w, optimal_tau_z, G)

    if converged:
        print("Solution status:", results["sol"].status)
        print("Solution message:", results["sol"].message)
        print("Solution vector [T_C, T_D, log(Z_C), log(Z_D), w_c, w_d, p_D, L]:")
        print(solution)

        print("\nProduction Summary:")
        print(f"Sector C: T_prod = {results['t_c']:.4f}, z_c = {results['z_c']:.4f}")
        print(f"Sector D: T_prod = {results['t_d']:.4f}, z_d = {results['z_d']:.4f}")
        print(f"Wage vector, wage = {results['wage_vector']}, p_D = {results['p_d']:.4f}")
        print(f"Sector C output, F_C = {results['f_c']:.4f}")
        print(f"Sector D output, F_D = {results['f_d']:.4f}")

        print("\nHousehold Demands and Leisure:")
        for i in range(n):
            print(f"Household {i+1}: c = {results['c_agents'][i]:.4f}, D = {results['d_agents'][i]:.4f}, l = {results['l_agents'][i]:.4f}")

        print("\nAggregated Quantities:")
        print(f"Aggregate c = {results['agg_c']:.4f}")
        print(f"Aggregate d = {results['agg_d']:.4f}")
        print(f"Aggregate labor supply = {results['agg_labor']:.4f}")

        print("\nLump Sum Transfer:")
        print(f"l = {results['l']:.4f}")

        print("\nFirm Profits:")
        print(f"Profit C: {results['profit_c']:.4f}")
        print(f"Profit D: {results['profit_d']:.4f}")

        print("\nHousehold Budget Constraints:")
        for i in range(n):
            print(f"Household {i+1}: error = {results['budget_errors'][i]:.10f}")

        print(f"\nGood C market clearing residual: {(results['agg_c'] + 0.5 * G) - results['f_c']:.10f}")
    else:
        print("Inner solver did not converge at optimal tax rates")

    agg_polluting = results['z_c'] + results['z_d']
    effective_utilities = results['utilities']
