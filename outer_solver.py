import numpy as np
from scipy.optimize import minimize
import inner_solver as solver

# Model parameters (consistent with inner_solver)
xi = 10.0
theta = 1.0
n = 5
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
G = 3.0  # Initial value for government spending

def maximize_welfare(G):
    """
    Maximizes social welfare by choosing optimal tax rates (tau_w, tau_z)
    given government spending G, using the inner solver.
    """

    def swf_obj(x):
        """
        Objective function: Negative of social welfare.
        """
        tau_w = x[0:n]
        tau_z = x[n]

        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return 1e10  # Penalize non-convergence

            # Extract relevant variables for welfare calculation
            utilities = results['utilities']  # Use the utilities from inner_solver
            agg_polluting = results['Z_C'] + results['Z_D']
            welfare = np.sum(utilities) - xi * (agg_polluting**theta)
            return -welfare  # Minimize the negative of welfare
        except Exception as e:
            print(f"Solver failed with error: {e}")
            return 1e10  # Return a large penalty value

    def ic_constraints(x):
        """
        Implements the Mirrlees IC constraints.
        """
        tau_w = x[0:n]
        tau_z = x[n]

        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return -np.ones(n*(n-1)) * 1e6  # Large negative penalty if not converged

            # Parameters for utility (consistent with inner_solver)
            alpha, beta, gamma = 0.7, 0.2, 0.2
            d0 = 5.5
            T = 100.0  # Time endowment

            # Compute income measure I for each type:
            I = np.zeros(n)
            for j in range(n):
                I[j] = (T - results['l_agents'][j])*(1.0 - tau_w[j])*phi[j]*results['w'] + results['L']

            g_list = []
            for i in range(n):
                U_i = results['utilities'][i]
                for j in range(n):
                    if i == j:
                        continue

                    c_j = results['C_agents'][j]
                    d_j = results['D_agents'][j]

                    denom = (1.0 - tau_w[j]) * phi[i] * results['w']
                    if denom <= 0:
                        g_list.append(-1e6)
                        continue

                    ell_i_j = T - I[j] / denom
                    if ell_i_j <= 0:
                        U_i_j = -1e6
                    else:
                        if c_j <= 0 or d_j <= d0:
                            U_i_j = -1e6
                        else:
                            U_i_j = (alpha * np.log(c_j) +
                                     beta * np.log(d_j - d0) +
                                     gamma * np.log(ell_i_j))

                    # Constraint: U_i must be at least U_i_j
                    g_list.append(U_i - U_i_j)

            return np.array(g_list)

        except Exception as e:
            print(f"IC constraint evaluation failed with error: {e}")
            return -np.ones(n*(n-1)) * 1e6  # Large negative penalty if solver fails

    # Initial guess for tax rates
    initial_tau_w = [0.05] * n
    initial_tau_z = 0.5
    initial_guess = np.array(initial_tau_w + [initial_tau_z])

    # Bounds for tax rates (adjust as needed)
    bounds = [(-2.0, 2.0)] * n + [(0.0001, 20.0)]

    # Constraints
    constraints = [
        {'type': 'ineq', 'fun': ic_constraints}  # IC constraints
    ]

    # Optimization
    result = minimize(swf_obj, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success:
        opt_tau_w = result.x[0:n]
        opt_tau_z = result.x[n]
        max_welfare = -result.fun

        print("Social Welfare Maximization Successful!")
        print("Optimal tau_w:", opt_tau_w)
        print("Optimal tau_z:", opt_tau_z)
        print("Maximized Social Welfare:", max_welfare)
        return opt_tau_w, opt_tau_z, max_welfare
    else:
        print("Optimization Failed!")
        print("Message:", result.message)
        return None, None, None

# Example usage:
if __name__ == "__main__":
    optimal_tau_w, optimal_tau_z, max_welfare = maximize_welfare(G)

    if optimal_tau_w is not None:
        print("\nDetailed Results at Optimal Tax Rates:")
        solution, results, converged = solver.solve(optimal_tau_w, optimal_tau_z, G)

        if converged:
            print("Solution status:", results["sol"].status)
            print("Solution message:", results["sol"].message)
            print("Solution vector [T_C, T_D, Z_C, Z_D, w, p_D, L]:")
            print(solution)

            print("\nProduction Summary:")
            print(f"Sector C: T_prod = {results['T_C']:.4f}, Z_C = {results['Z_C']:.4f}")
            print(f"Sector D: T_prod = {results['T_D']:.4f}, Z_D = {results['Z_D']:.4f}")
            print(f"Common wage, w = {results['w']:.4f}, p_D = {results['p_D']:.4f}")
            print(f"Sector C output, F_C = {results['F_C']:.4f}")
            print(f"Sector D output, F_D = {results['F_D']:.4f}")

            print("\nHousehold Demands and Leisure:")
            for i in range(n):
                print(f"Household {i+1}: C = {results['C_agents'][i]:.4f}, D = {results['D_agents'][i]:.4f}, l = {results['l_agents'][i]:.4f}")

            print("\nAggregated Quantities:")
            print(f"Aggregate C = {results['agg_C']:.4f}")
            print(f"Aggregate D = {results['agg_D']:.4f}")
            print(f"Labor supply = {results['agg_labor']:.4f}")

            print("\nLump sum:")
            print(f"L = {results['L']:.4f}")

            print("\nFirm Profits:")
            print(f"Profit Sector C: {results['profit_C']:.4f}")
            print(f"Profit Sector D: {results['profit_D']:.4f}")

            print("\nHousehold Budget Constraints:")
            for i in range(n):
                print(f"Household {i+1}: Error = {results['budget_errors'][i]:.10f}")

            print(f"\nGood C Market Clearing Residual: {(results['agg_C'] + 0.5*G) - results['F_C']:.10f}")
        else:
            print("Inner solver did not converge at optimal tax rates.")