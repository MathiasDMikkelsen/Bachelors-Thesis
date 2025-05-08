import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
import j_inner_solver_ext as solver
from j_inner_solver_ext import alpha, beta, gamma, d0, phi, varphi, n, p_a, t as T

# a. maximize social welfare
def maximize_welfare(G, xi, p_a, varsigma):
    
    # a.1 define the objective function, which returns the negative of social welfare for current guess (x)
    def swf_obj(x):
        
        # set policy vector to current guess
        tau_w = x[0:n]
        tau_z = x[n]
        
        # solve for general equilibrium (ge) and retrieve results
        solution, results, converged = solver.solve(tau_w, tau_z, G, p_a, varsigma)
        
        # penalize inner layer non-convergence
        if not converged:
            return 1e10 
        
        # retrieve utilities and aggregare pollution for ge solution
        utilities = results['utilities']
        agg_polluting = results['z_c'] + results['z_d']
        
        # compute and return negative social welfare for current guess
        theta = 1.0
        welfare = np.sum(utilities) - 5 * xi * (agg_polluting**theta)
        return -welfare

    # a.2 define the ic-constraints and compute violations for current guess (x)
    def ic_constraints(x):
        
        # set policy vector to current guess
        tau_w = x[0:n]
        tau_z = x[n]
        
        # solve for general equilbrium and retrive results. penalize if the inner solver did not converge
        solution, results, converged = solver.solve(tau_w, tau_z, G, p_a, varsigma)
        if not converged:
            return -np.ones(n*(n-1)) * 1e6
            
        # compute income for each household
        I = np.zeros(n)
        for j in range(n):
            I[j] = (T - results['ell_agents'][j]) * (1.0 - tau_w[j]) * phi[j] * results['w']+(1-tau_w[j])*varphi[j]*(results['a_c']+results['a_d'])*p_a
            
        # define constraint function value array
        g_list = []
        
            # compute value of each constraint function
        for i in range(n):
                
            # retrieve blue log utility for household i if it provides correct productivity
            U_i = results['utilities'][i]
                
            # compute blue log utility for household i if it pretends to be household j
            for j in range(n):
                    
                # ic-constraint always holds if household i pretends to be itself. continue in this case
                if i == j:
                    continue
                    
                # retrieve consumption of both goods and labor for household j
                c_j = results['c_agents'][j]
                d_j = results['d_agents'][j]
                    
                # compute after tax wage per labor unit for household j. penalize if infeasible
                denom = (1.0 - tau_w[j]) * phi[i] * results['w']
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
                    U_i_j = (
                        alpha * np.log(c_j) +
                        beta * np.log(d_j - d0) +
                         gamma * np.log(ell_i_j)
                    )
                    
                # append constraint function value to array
                g_list.append(U_i - U_i_j)
            
        # return constraint function values array
        return np.array(g_list)

    # a.3 define the nonlinear constraint. constaint is violated if household i is strictly better of pretending to be household j, hence lower bound 0 and upper infinity
    nonlinear_constraint = NonlinearConstraint(ic_constraints, lb=0, ub=np.inf)

    # a.4 define the initial guess for optimization
    initial_tau_w = [-2.5, -0.5, -0.2, 0.1, 0.5]
    initial_tau_z = 0.5
    initial_guess = np.array(initial_tau_w + [initial_tau_z])

     # a.5 define optimization bounds
    bounds = [(-10.0, 10.0)] * n + [(1e-6, 100.0)]

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
        opt_tau_w = result.x[0:n]
        opt_tau_z = result.x[n]
        max_welfare = -result.fun
        print("social welfare maximization successful")
        print("optimal tau_w:", opt_tau_w)
        print("optimal tau_z:", opt_tau_z)
        print("maximized social welfare:", max_welfare)
        return opt_tau_w, opt_tau_z, max_welfare
    else:
        print("optimization failed")
        print("message:", result.message)
        return None, None, None

# b. test the solver

# b.1 parameters
G = 5.0
p_a = 5.0
xi_example_value = 0.1
varsigma = 2.0
optimal_tau_w, optimal_tau_z, max_welfare = maximize_welfare(G, xi_example_value, p_a, varsigma)

# b.2 solve for optimal policy and print results 
if optimal_tau_w is not None:
    
    print("\nresults at optimal tax rates:")
    solution, results, converged = solver.solve(optimal_tau_w, optimal_tau_z, G, p_a, varsigma)

    if converged:

        sol_obj = results.get('sol', None)
        if sol_obj is not None:
            print("solution status:", sol_obj.status)
            print("solution message:", sol_obj.message)
        else:
            print("solution object not returned by solver.")

        print("solution vector [t_c, t_d, log_z_c, log_z_d, log_a_c, log_a_d, w, p_d, l]:")
        print(solution)

        # production summary
        print("\nproduction summary:")
        print(f"sector C: t_c = {results['t_c']:.4f}, z_c = {results['z_c']:.4f}, a_c = {results['a_c']:.4f}")
        print(f"sector D: t_d = {results['t_d']:.4f}, z_d = {results['z_d']:.4f}, a_d = {results['a_d']:.4f}")
        print(f"common wage, w = {results['w']:.4f}, p_d = {results['p_d']:.4f}")
        print(f"sector C output, f_c = {results['f_c']:.4f}")
        print(f"sector D output, f_d = {results['f_d']:.4f}")

        # household demands and leisure
        print("\nhousehold demands and leisure:")
        for i in range(n):
            print(f"household {i+1}: c = {results['c_agents'][i]:.4f}, d = {results['d_agents'][i]:.4f}, l = {results['ell_agents'][i]:.4f}")

        # aggregates
        print("\naggregate auantities:")
        print(f"aggregate c = {results['agg_c']:.4f}")
        print(f"aggregate d = {results['agg_d']:.4f}")
        print(f"aggregate labor supply = {results['agg_labor']:.4f}")

        # lump sum 
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