import numpy as np
from scipy.optimize import minimize
import inner_solver as solver

# Model parameters (consistent with inner_solver)
xi = 0.2
theta = 1.0
n = 5
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
G = 5.0  # Initial value for government spending
T_endowment = 100.0  # Time endowment
d0 = 5.0           # Subsistence level for polluting consumption

def maximize_welfare(G):

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
            utilities = results['utilities']  # Extract household utilities
            agg_polluting = results['Z_C'] + results['Z_D']
            welfare = np.sum(utilities) - xi * (agg_polluting**theta)
            return -welfare  # Minimization: return negative welfare
        except Exception as e:
            print(f"Solver failed with error: {e}")
            return 1e10

    def ic_constraints(x):
        """
        Implements the Mirrlees IC constraints.
        For each household pair, the constraint is:
            U_i - U_i^j >= 0,
        where U_i is the actual utility, and U_i^j is the "pretend" utility.
        """
        tau_w = x[0:n]
        tau_z = x[n]
        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return -np.ones(n*(n-1)) * 1e6
            # Utility parameters
            alpha, beta, gamma = 0.7, 0.2, 0.2
            T = T_endowment

            # Compute income measure I for each household
            I = np.zeros(n)
            for j in range(n):
                I[j] = (T - results['l_agents'][j]) * (1.0 - tau_w[j]) * phi[j] * results['w'] + results['L']

            g_list = []
            # For each household i and each alternative j:
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
                            U_i_j = c_j**alpha * (d_j - d0)**beta * ell_i_j**gamma
                    # Constraint: U_i must be at least U_i_j
                    g_list.append(U_i - U_i_j)
            return np.array(g_list)
        except Exception as e:
            print(f"IC constraint evaluation failed with error: {e}")
            return -np.ones(n*(n-1)) * 1e6

    def constraint_positive_Z(x):
        """
        Ensure that both Z_C and Z_D are strictly positive.
        Returns min(Z_C, Z_D) - tol so that the constraint is satisfied only if both exceed tol.
        """
        tau_w = x[0:n]
        tau_z = x[n]
        tol = 1e-6
        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return -1e6
            Z_C = results['Z_C']
            Z_D = results['Z_D']
            # Convert to scalars if needed
            if isinstance(Z_C, (list, np.ndarray)):
                Z_C = np.array(Z_C).flatten()[0]
            if isinstance(Z_D, (list, np.ndarray)):
                Z_D = np.array(Z_D).flatten()[0]
            return min(Z_C, Z_D) - tol
        except Exception as e:
            print(f"Positive Z constraint evaluation error: {e}")
            return -1e6

    # Initial guess for tax rates
    initial_tau_w = [-2.5, -0.5, 0.0, 0.5, 0.5]
    initial_tau_z = 1.0
    initial_guess = np.array(initial_tau_w + [initial_tau_z])

    # Bounds for tax rates: n bounds for tau_w and one for tau_z
    bounds = [(-2.5, 1.5)] * n + [(0.0001, 20.0)]

    # Constraints: include both IC constraints and positive Z constraint
    constraints = [
        {'type': 'ineq', 'fun': ic_constraints},
        {'type': 'ineq', 'fun': constraint_positive_Z}
    ]

    # Run the optimization using trust-constr
    result = minimize(swf_obj, initial_guess, method='trust-constr', bounds=bounds, constraints=constraints)

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

if __name__ == "__main__":
    optimal_tau_w, optimal_tau_z, max_welfare = maximize_welfare(G)
    if optimal_tau_w is not None:
        print("\nDetailed Results at Optimal Tax Rates:")
        solution, results, converged = solver.solve(optimal_tau_w, optimal_tau_z, G)
        if converged:
            print("Solution vector:")
            print(solution)
            print("\nResults:")
            for key, value in results.items():
                print(f"{key}: {value}")
        else:
            print("Inner solver did not converge at optimal tax rates.")