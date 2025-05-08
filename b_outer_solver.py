import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import a_inner_solver as solver
from a_inner_solver import alpha, beta, gamma, phi, n, t, d0, r, t as T, g as G

# a. maximize social welfare
def maximize_welfare(G, xi, r, d0):

    # a.1 define the objective function, which returns the negative of social welfare for current guess (x)
    def swf_obj(x):
        
        # set policy vector to current guess
        tau_w, tau_z = x[:n], x[n] 
        
        # solve for general equilibrium (ge) and retrieve results
        _, results, conv = solver.solve(tau_w, tau_z, G, r, d0)
        
        # penalize if the inner solver did not converge
        if not conv:
            return 1e10
        
        # retrieve log blue utilities and aggregare pollution for ge solution
        utils = results['utilities']
        agg_z = results['z_c'] + results['z_d']
        
        # compute and return negative social welfare for current guess
        return -(np.sum(utils) - 5 * xi * (agg_z**theta))

    # a.2 define the ic-constraints and compute violations for current guess (x)
    def ic_constraints(x):
        
        # set policy vector to current guess
        tau_w, tau_z = x[:n], x[n]
        
        # solve for general equilbrium and retrive results. penalize if the inner solver did not converge
        _, results, conv = solver.solve(tau_w, tau_z, G, r, d0)
        if not conv:
            return -np.ones(n*(n-1)) * 1e6

        # compute income for each household
        I = np.array([
            (T - results['l_agents'][j]) * (1 - tau_w[j]) * phi[j] * results['w']
            for j in range(n)
        ])

        # define constraint function value array
        g_list = []
        
        # compute value of each constraint function
        for i in range(n):
            
            # retrieve log blue utility for household i if it provides correct productivity
            U_i = results['utilities'][i]
            
            # compute log blue utility for household i if it pretends to be household j
            for j in range(n):
                
                # ic-constraint always holds if household i pretends to be itself. continue in this case
                if i == j:
                    continue
                
                # retrieve consumption of both goods and labor for household j
                c_j   = results['c_agents'][j]
                d_j   = results['d_agents'][j]
                
                # compute after tax wage per labor unit for household j
                denom = (1 - tau_w[j]) * phi[i] * results['w']
                
                # penalize if denominator is zero
                if denom == 0:
                    g_list.append(-1e6)
                    continue
                
                # compute leisure for household i if it pretends to be household j
                ell_i_j = T - I[j] / denom
                
                # penalize if leisure is negative
                if ell_i_j <= 0 or c_j <= 0 or d_j <= d0:
                    U_i_j = -1e6
                
                # if leisure is positive, compute log blue utility for household i if it pretends to be household j
                else:
                    U_i_j = (alpha * np.log(c_j) +
                             beta  * np.log(d_j - d0) +
                             gamma * np.log(ell_i_j))
                
                # append constraint function value to array
                g_list.append(U_i - U_i_j)
        
        # return constraint function values array       
        return np.array(g_list)

    # a.3 define the nonlinear constraint. constaint is violated if household i is strictly better of pretending to be household j, hence lower bound 0 and upper infinity
    nonlinear_constraint = NonlinearConstraint(ic_constraints, lb=0, ub=np.inf)

    # a.4 define the initial guess for optimization
    initial_guess = [-2.5, -0.5, -0.2, 0.1, 0.5, 0.5]  # initial_guess = [np.zeros(n), 0.5] sometimes works better
    
    # a.5 define optimization bounds
    bounds = [(-4.0, 10.0)] * n + [(1e-6, 100.0)] 

    # a.6 minimize negative social welfare using slsqp
    result = minimize(
        swf_obj,
        initial_guess,
        method='SLSQP',
        bounds=bounds,
        constraints=[nonlinear_constraint]
    )

    # a.7 retrieve results and return them + print some info
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

# b. test the solver

# b.1 additional test parameters (other test parameters are import from inner_solver)
xi = 0.1
theta = 1.0

# b.2 optimize social welfare
opt_tau_w, opt_tau_z, max_welf = maximize_welfare(G, xi, r, d0)

# b.3 solve for be at optimal policy and print results
if opt_tau_w is not None:
    sol_vec, results, conv = solver.solve(opt_tau_w, opt_tau_z, G, r, d0)
    if conv:
        
        # print basic solution info
        print("\nsolution status:", results["sol"].status)
        print("solution message:", results["sol"].message)
        print("solution vector [t_c, t_d, log_z_c, log_z_d, w, p_d, l]:")
        print(sol_vec)

        # print all residuals
        residuals = results["system_residuals"]
        for idx, val in enumerate(residuals, start=1):
            print(f"eq{idx} residual: {val:.6e}")

        # government budget eq7 residual
        print(f"gov budget residual (eq7): {residuals[6]:.6e}")

        # aggregate pollution
        agg_z = results['z_c'] + results['z_d']
        print(f"aggregate pollution (z_c+z_d): {agg_z:.4f}")

    else:
        print("inner solver did not converge at optimal tax rates")