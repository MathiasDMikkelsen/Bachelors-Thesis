import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import inner_solver_ext as solver
from inner_solver_ext import alpha, beta, gamma, d0, phi, varphi, n, p_a

# a. parameters
T = 24
# xi parameter for pollution penalty is now passed into maximize_welfare
theta = 1.0  # exponent on aggregate pollution in welfare penalty

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
            agg_polluting = results['z_c'] + results['z_d']
            welfare = np.sum(utilities) - 5 * xi * (agg_polluting**theta)
            return -welfare
        except Exception:
            return 1e10

    # b2. define ic constraints
    def ic_constraints(x):
        tau_w = x[0:n]
        tau_z = x[n]
        try:
            solution, results, converged = solver.solve(tau_w, tau_z, G)
            if not converged:
                return -np.ones(n*(n-1)) * 1e6
            I = np.zeros(n)
            for j in range(n):
                I[j] = (T - results['l_agents'][j]) * (1.0 - tau_w[j]) * phi[j] * results['w']+(1-tau_w[j])*varphi[j]*(results['a_c']+results['a_d'])*p_a
            g_list = []
            for i in range(n):
                U_i = results['utilities'][i]
                for j in range(n):
                    if i == j:
                        continue
                    c_j = results['c_agents'][j]
                    d_j = results['d_agents'][j]
                    denom = (1.0 - tau_w[j]) * phi[i] * results['w']
                    if denom == 0:
                        g_list.append(-1e6)
                        continue
                    ell_i_j = T - I[j] / denom
                    if ell_i_j <= 0 or c_j <= 0 or d_j <= d0:
                        U_i_j = -1e6
                    else:
                        U_i_j = (
                            alpha * np.log(c_j) +
                            beta * np.log(d_j - d0) +
                            gamma * np.log(ell_i_j)
                        )
                    g_list.append(U_i - U_i_j)
            return np.array(g_list)
        except Exception:
            return -np.ones(n*(n-1)) * 1e6

    # b3. nonlinear constraints for incentive compatibility
    nonlinear_constraint = NonlinearConstraint(ic_constraints, lb=0, ub=np.inf)

    # b4. initial guess for taxes
    initial_tau_w = [-2.5, -0.5, -0.2, 0.1, 0.5]
    initial_tau_z = 0.5
    initial_guess = np.array(initial_tau_w + [initial_tau_z])

    # b5. bounds for tax rates
    bounds = [(-10.0, 10.0)] * n + [(1e-6, 100.0)]

    # b6. minimize negative welfare
    result = minimize(
        swf_obj,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=[nonlinear_constraint]
    )

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
if __name__ == "__main__":
    G = 5.0
    xi_example_value = 0.1
    optimal_tau_w, optimal_tau_z, max_welfare = maximize_welfare(G, xi_example_value)

    if optimal_tau_w is not None:
        print("\nresults at optimal tax rates:")
        solution, results, converged = solver.solve(optimal_tau_w, optimal_tau_z, G)

        if converged:
            # Retain printing solution status/message even if 'sol' is nested under results
            sol_obj = results.get('sol', None)
            if sol_obj is not None:
                print("solution status:", sol_obj.status)
                print("solution message:", sol_obj.message)
            else:
                print("solution object not returned by solver.")

            print("solution vector [t_c, t_d, log_z_c, log_z_d, log_a_c, log_a_d, w, p_d, l]:")
            print(solution)

            # production summary
            print("\nproduction Summary:")
            print(f"sector C: t_c = {results['t_c']:.4f}, z_c = {results['z_c']:.4f}, a_c = {results['a_c']:.4f}")
            print(f"sector D: t_d = {results['t_d']:.4f}, z_d = {results['z_d']:.4f}, a_d = {results['a_d']:.4f}")
            print(f"common wage, w = {results['w']:.4f}, p_d = {results['p_d']:.4f}")
            print(f"sector C output, f_c = {results['f_c']:.4f}")
            print(f"sector D output, f_d = {results['f_d']:.4f}")

            # household Demands and Leisure
            print("\nhousehold Demands and Leisure:")
            for i in range(n):
                print(f"household {i+1}: c = {results['c_agents'][i]:.4f}, d = {results['d_agents'][i]:.4f}, l = {results['l_agents'][i]:.4f}")

            # aggregated Quantities
            print("\naggregated Quantities:")
            print(f"aggregate c = {results['agg_c']:.4f}")
            print(f"aggregate d = {results['agg_d']:.4f}")
            print(f"aggregate labor supply = {results['agg_labor']:.4f}")

            # lump sum l
            print("\nlump sum:")
            print(f"l = {results['l']:.4f}")

            # firm profits
            print("\nfirm profits:")
            print(f"profit c: {results['profit_c']:.4f}")
            print(f"profit d: {results['profit_d']:.4f}")

            # household budget constraints
            print("\nhousehold budget constraints:")
            for i in range(n):
                print(f"household {i+1}: error = {results['budget_errors'][i]:.10f}")

            # good c market clearing residual
            print(f"\ngood c market clearing residual: {(results['agg_c'] + 0.5*G) - results['f_c']:.10f}")
        else:
            print("inner solver did not converge at optimal tax rates")

        # aggregate polluting and utilities for further use
        agg_polluting = results['z_c'] + results['z_d']
        effective_utilities = results['utilities']
    
    # end of __main__
