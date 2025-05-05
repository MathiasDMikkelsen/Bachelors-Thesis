import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import inner_solver as solver
from inner_solver import alpha, beta, gamma, phi, n, epsilon_c, epsilon_d

# a. parameters
T = 24.0
theta = 1.0
G = 5.0

def maximize_welfare(G, xi, r, d0):
    """
    maximize social welfare given:
      - G:   government spending
      - xi:  environmental preference parameter
      - r:   ces exponent
      - d0:  baseline consumption of good d
    returns: optimal (tau_w array, tau_z scalar, max_welfare)
    """

    def swf_obj(x):
        tau_w, tau_z = x[:n], x[n]
        _, results, conv = solver.solve(tau_w, tau_z, G, r, d0)
        if not conv:
            return 1e10
        utils = results['utilities']
        agg_z = results['z_c'] + results['z_d']
        return -(np.sum(utils) - 5 * xi * (agg_z**theta))

    def ic_constraints(x):
        tau_w, tau_z = x[:n], x[n]
        _, results, conv = solver.solve(tau_w, tau_z, G, r, d0)
        if not conv:
            return -np.ones(n*(n-1)) * 1e6

        I = np.array([
            (T - results['l_agents'][j]) * (1 - tau_w[j]) * phi[j] * results['w']
            for j in range(n)
        ])

        g_list = []
        for i in range(n):
            U_i = results['utilities'][i]
            for j in range(n):
                if i == j:
                    continue
                c_j   = results['c_agents'][j]
                d_j   = results['d_agents'][j]
                denom = (1 - tau_w[j]) * phi[i] * results['w']
                if denom == 0:
                    g_list.append(-1e6)
                    continue
                ell_i_j = T - I[j] / denom
                if ell_i_j <= 0 or c_j <= 0 or d_j <= d0:
                    U_i_j = -1e6
                else:
                    U_i_j = (alpha * np.log(c_j) +
                             beta  * np.log(d_j - d0) +
                             gamma * np.log(ell_i_j))
                g_list.append(U_i - U_i_j)
        return np.array(g_list)

    nonlinear_constraint = NonlinearConstraint(ic_constraints, lb=0, ub=np.inf)

    #initial_guess = np.hstack((np.zeros(n), 0.5))
    initial_guess = [-2.5, -0.5, -0.2, 0.1, 0.5, 0.5]
    bounds = [(-10.0, 10.0)] * n + [(1e-6, 100.0)]

    result = minimize(
        swf_obj,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=[nonlinear_constraint]
    )

    if result.success:
        opt_tau_w   = result.x[:n]
        opt_tau_z   = result.x[n]
        max_welf    = -result.fun
        print("social welfare maximization successful")
        print("optimal tau_w:", opt_tau_w)
        print("optimal tau_z:", opt_tau_z)
        print("maximized social welfare:", max_welf)
        return opt_tau_w, opt_tau_z, max_welf
    else:
        print("optimization failed")
        print("message:", result.message)
        return None, None, None

# c. run optimization for a test xi with r and d0
xi_example = 0.1
r_val       = -1.0
d0_val      = 0.5

opt_tau_w, opt_tau_z, max_welf = maximize_welfare(G, xi_example, r_val, d0_val)

if opt_tau_w is not None:
    sol_vec, results, conv = solver.solve(opt_tau_w, opt_tau_z, G, r_val, d0_val)
    if conv:
        # print basic solution info
        print("\nsolution status:", results["sol"].status)
        print("solution message:", results["sol"].message)
        print("solution vector [t_c, t_d, log_z_c, log_z_d, w, p_d, l]:")
        print(sol_vec)

        # --- print all residuals ---
        residuals = results["system_residuals"]
        for idx, val in enumerate(residuals, start=1):
            print(f"eq{idx} residual: {val:.6e}")

        # government budget eq7 residual
        print(f"gov budget residual (eq7): {residuals[6]:.6e}")

        # firm c condition
        w   = results['w']
        tau_z = opt_tau_z
        expected_c = (((1 - epsilon_c) * w) / (tau_z * epsilon_c)) ** (1/(r_val-1))
        comp_c     = results['t_c'] / results['z_c']
        print(f"firm c: t_c/z_c = {comp_c:.6e}, expected = {expected_c:.6e}, residual = {(comp_c-expected_c):.6e}")

        # firm d condition
        expected_d = (((1 - epsilon_d) * w) / (tau_z * epsilon_d)) ** (1/(r_val-1))
        comp_d     = results['t_d'] / results['z_d']
        print(f"firm d: t_d/z_d = {comp_d:.6e}, expected = {expected_d:.6e}, residual = {(comp_d-expected_d):.6e}")

        # aggregate pollution
        agg_z = results['z_c'] + results['z_d']
        print(f"aggregate pollution (z_c+z_d): {agg_z:.4f}")

    else:
        print("inner solver did not converge at optimal tax rates")