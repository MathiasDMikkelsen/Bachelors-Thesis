import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import inner_solver as solver

# Model parameters (consistent with inner_solver)
xi = 0.1
theta = 1.0
n = 5
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
G = 5.0  # Initial value for government spending

def maximize_welfare(G):

    def swf_obj(x):

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
  
        tau_w = x[0:n]
        tau_z = x[n]

        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return -np.ones(n*(n-1)) * 1e6  # Large negative penalty if not converged

            # Parameters for utility (consistent with inner_solver)
            alpha, beta, gamma = 0.7, 0.2, 0.2
            d0 = 0.5
            T = 24.0  # Time endowment

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
                    g_list.append(U_i - U_i_j) # Add small tolerance to avoid strict inequality

            return np.array(g_list)

        except Exception as e:
            print(f"IC constraint evaluation failed with error: {e}")
            return -np.ones(n*(n-1)) * 1e6  # Large negative penalty if solver fails

    # Define the nonlinear constraint
    nonlinear_constraint = NonlinearConstraint(ic_constraints, lb=0, ub=np.inf)

    # Initial guess for tax rates
    #initial_tau_w = [-2.5, -0.5, -0.2, 0.1, 0.5]
    initial_tau_w = [(0.0)]*n # INITIAL GUESSES DO NOT MATTER! KLENERT GO HOME!!!!!!!!!
    initial_tau_z = 0.5
    initial_guess = np.array(initial_tau_w + [initial_tau_z])

    # Bounds for tax rates (adjust as needed)
    bounds = [(-10.0, 10.0)] * n + [(1e-6, 100.0)]

    # Optimization using trust-constr
    result = minimize(swf_obj, initial_guess, method='SLSQP', bounds=bounds, constraints=[nonlinear_constraint])

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

# Example usage:
if __name__ == "__main__":
    optimal_tau_w, optimal_tau_z, max_welfare = maximize_welfare(G)

    if optimal_tau_w is not None:
        print("\nresults at optimal tax rates:")
        solution, results, converged = solver.solve(optimal_tau_w, optimal_tau_z, G)

        if converged:
            print("solution status:", results["sol"].status)
            print("solution message:", results["sol"].message)
            print("solution vector [T_C, T_D, Z_C, Z_D, w, p_D, L]:")
            print(solution)

            print("\nproduction Summary:")
            print(f"sector C: T_prod = {results['T_C']:.4f}, Z_C = {results['Z_C']:.4f}")
            print(f"sector D: T_prod = {results['T_D']:.4f}, Z_D = {results['Z_D']:.4f}")
            print(f"common wage, w = {results['w']:.4f}, p_D = {results['p_D']:.4f}")
            print(f"sector C output, F_C = {results['F_C']:.4f}")
            print(f"sector D output, F_D = {results['F_D']:.4f}")

            print("\nhousehold Demands and Leisure:")
            for i in range(n):
                print(f"household {i+1}: C = {results['C_agents'][i]:.4f}, D = {results['D_agents'][i]:.4f}, l = {results['l_agents'][i]:.4f}")

            print("\naggregated Quantities:")
            print(f"aggregate C = {results['agg_C']:.4f}")
            print(f"aggregate D = {results['agg_D']:.4f}")
            print(f"labor supply = {results['agg_labor']:.4f}")

            print("\nlump sum:")
            print(f"L = {results['L']:.4f}")

            print("\nfirm profits:")
            print(f"profit Sector C: {results['profit_C']:.4f}")
            print(f"profit Sector D: {results['profit_D']:.4f}")

            print("\nhousehold budget constraints:")
            for i in range(n):
                print(f"household {i+1}: error = {results['budget_errors'][i]:.10f}")

            print(f"\ngood C market clearing residual: {(results['agg_C'] + 0.5*G) - results['F_C']:.10f}")
        else:
            print("inner solver did not converge at optimal tax rates")