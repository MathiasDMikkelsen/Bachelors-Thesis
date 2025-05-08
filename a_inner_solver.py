import numpy as np
from scipy.optimize import root

# a. parameters
alpha     = 0.7
beta      = 0.2
gamma     = 0.2
epsilon_c = 0.995
epsilon_d = 0.92
p_c       = 1.0
phi       = np.array([0.03, 0.0825, 0.141, 0.229, 0.511])
n         = len(phi)
t         = 24.0  # time endowment

# b. solve for general equilibirum
def solve(tau_w, tau_z, g, r, d0):

    # b.1 form system of equations
    def system_eqns(y):
        
        # unknowns vector
        t_c, t_d, log_z_c, log_z_d, w, p_d, l = y 
        
        # transformations
        z_c = np.exp(log_z_c)
        z_d = np.exp(log_z_d)

        # define production
        f_c = (epsilon_c * t_c**r + (1 - epsilon_c) * z_c**r)**(1/r)
        f_d = (epsilon_d * t_d**r + (1 - epsilon_d) * z_d**r)**(1/r)

        # calculate demands for dirty good and leisure
        common = phi * w * (1 - tau_w) * t + l - p_d * d0
        d_agents = beta / (p_d*(alpha+beta+gamma)) * common + d0
        l_agents = gamma / ((alpha+beta+gamma)*(1 - tau_w)*phi*w) * common

        # calculate aggregare labor supply and aggregare dirty good demand
        agg_labor = np.sum(phi * (t - l_agents))
        agg_d     = np.sum(d_agents)

        # define system equations
        eq1 = t_c + t_d - agg_labor # labor market cleating
        eq2 = (agg_d + 0.5*g/p_d) - f_d # dirty good market clearing
        
        eq3 = w - epsilon_c * t_c**(r-1) * f_c**(1-r)  # firm c labor foc
        eq4 = tau_z - (1 - epsilon_c) * z_c**(r-1) * f_c**(1-r) # firm c pollution foc
        
        eq5 = w - epsilon_d * t_d**(r-1) * f_d**(1-r) * p_d # firm d labor foc
        eq6 = tau_z - (1 - epsilon_d) * z_d**(r-1) * f_d**(1-r) * p_d # firm d pollution foc
        
        eq7 = n*l - (np.sum(tau_w * w * phi * (t - l_agents)) +  tau_z*(z_c + z_d) - g) # gov't budget constrint

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7]) # form equation matrix

    # b.2 initial guess
    y0 = np.array([0.3, 0.4, np.log(0.6), np.log(0.4), 0.5, 1.5, 0.1])
    
    # b.3 solve system of equations for unknowns vector using lm
    sol = root(system_eqns, y0, method='lm')

    # b.4 unpack solution
    t_c, t_d, log_z_c, log_z_d, w, p_d, l = sol.x
    z_c, z_d = np.exp(log_z_c), np.exp(log_z_d)

    # b.5 compute equilibrium output
    f_c = (epsilon_c * t_c**r + (1 - epsilon_c) * z_c**r)**(1/r)
    f_d = (epsilon_d * t_d**r + (1 - epsilon_d) * z_d**r)**(1/r)

    # b.6 compute equilibirum allocations for all households
    common = phi * w * (1 - tau_w) * t + l - p_d * d0
    c_agents = alpha / (p_c*(alpha+beta+gamma)) * common
    d_agents = beta / (p_d*(alpha+beta+gamma)) * common + d0
    l_agents = gamma / ((alpha+beta+gamma)*(1 - tau_w)*phi*w) * common

    # b.7 compute equilibrium aggregates
    agg_c     = np.sum(c_agents)
    agg_d     = np.sum(d_agents)
    agg_labor = np.sum(phi * (t - l_agents))
    
    # b.8 compute equilibirum profits
    profit_c  = f_c - w*t_c - tau_z*z_c
    profit_d  = p_d*f_d - w*t_d - tau_z*z_d

    # b.9 compute budget errors
    budget_errors = np.zeros(n)
    for i in range(n):
        income      = phi[i]*w*(1 - tau_w[i])*(t - l_agents[i]) + l
        expenditure = c_agents[i] + p_d*d_agents[i]
        budget_errors[i] = income - expenditure

    # b.10 compute log blue utilities for all households
    utilities = np.zeros(n)
    for i in range(n):
        if c_agents[i] > 0 and (d_agents[i] - d0) > 0 and l_agents[i] > 0:
            utilities[i] = (alpha*np.log(c_agents[i]) +
                            beta*np.log(d_agents[i] - d0) +
                            gamma*np.log(l_agents[i]))
        else:
            utilities[i] = -1e6

    # b.11 results
    results = {
        "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d,
        "w": w, "p_d": p_d, "l": l,
        "f_c": f_c, "f_d": f_d,
        "c_agents": c_agents, "d_agents": d_agents, "l_agents": l_agents,
        "agg_c": agg_c, "agg_d": agg_d, "agg_labor": agg_labor,
        "profit_c": profit_c, "profit_d": profit_d,
        "budget_errors": budget_errors,
        "utilities": utilities,
        "sol": sol,
        "system_residuals": system_eqns(sol.x)
    }

    # b.12 return results
    return sol.x, results, sol.success

# c. test the solver

# c.1 test parameters
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_z = 4.0
g     = 5.0
r     = -1.0  
d0    = 0.5   

# c.2 solve for general equilibirum
solution, results, converged = solve(tau_w, tau_z, g, r, d0)

# c.3 print results
print("\nsolution status:", results["sol"].status)
print("solution message:", results["sol"].message)
print("converged:", converged)
print("solution vector [t_c, t_d, log_z_c, log_z_d, w, p_d, l]:")
print(solution)

print(f"\nsector c: t_c = {results['t_c']:.4f}, z_c = {results['z_c']:.4f}")
print(f"sector d: t_d = {results['t_d']:.4f}, z_d = {results['z_d']:.4f}")

print("\nhousehold demands and leisure:")
for i in range(n):
    print(f"  hh {i+1}: c = {results['c_agents'][i]:.4f}, "
          f"d = {results['d_agents'][i]:.4f}, "
          f"l = {results['l_agents'][i]:.4f}")

print("\naggregate results:")
print(f"c = {results['agg_c']:.4f}")
print(f"d = {results['agg_d']:.4f}")
print(f"l = {results['agg_labor']:.4f}")
print(f"u = {np.sum(results['utilities']):.4f}")
print(f"z = {results['z_c'] + results['z_d']:.4f}")