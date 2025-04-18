import numpy as np
from scipy.optimize import root

# a. parameters
alpha = 0.7
beta = 0.2
gamma = 0.2
r = -1.0
t = 24.0     
d0 = 0.5
epsilon_c = 0.995     
epsilon_d = 0.92   
p_c = 1.0      
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.511])
#phi = np.array([0.2]*5)
n = len(phi)

def solve(tau_w, tau_z, g):
    
    def system_eqns(y):
        t_c, t_d, log_z_c, log_z_d, w, p_d, l = y
        z_c = np.exp(log_z_c)
        z_d = np.exp(log_z_d)
        
        f_c = (epsilon_c * (t_c**r) + (1 - epsilon_c) * (z_c**r))**(1/r)
        f_d = (epsilon_d * (t_d**r) + (1 - epsilon_d) * (z_d**r))**(1/r)
        
        d_agents = (beta/(p_d*(alpha+beta+gamma))) * (phi*w*(1-tau_w)*t + l - p_d*d0) + d0
        l_agents = (gamma/((alpha+beta+gamma)*(1-tau_w)*phi*w)) * (phi*w*(1-tau_w)*t + l - p_d*d0)
        
        agg_labor = np.sum(phi*(t - l_agents))
        agg_d = np.sum(d_agents)
        
        eq1 = t_c + t_d - agg_labor
        eq2 = (agg_d + 0.5*g/p_d) - f_d
        eq3 = w - epsilon_c * (t_c**(r-1)) * (f_c**(1-r))
        eq4 = tau_z - (1 - epsilon_c) * (np.exp(log_z_c)**(r-1)) * (f_c**(1-r))
        eq5 = w - epsilon_d * (t_d**(r-1)) * (f_d**(1-r)) * p_d
        eq6 = tau_z - (1 - epsilon_d) * (np.exp(log_z_d)**(r-1)) * (f_d**(1-r)) * p_d
        eq7 = n*l - (np.sum(tau_w * w * phi * (t - l_agents)) + tau_z*(z_c+z_d) - g)
        
        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7])
    
    y0 = np.array([0.3, 0.4, np.log(0.6), np.log(0.4), 0.5, 1.5, 0.1])
    sol = root(system_eqns, y0, method='lm')
    
    t_c, t_d, log_z_c, log_z_d, w, p_d, l = sol.x
    z_c = np.exp(log_z_c)
    z_d = np.exp(log_z_d)
    
    f_c = (epsilon_c * (t_c**r) + (1 - epsilon_c) * (z_c**r))**(1/r)
    f_d = (epsilon_d * (t_d**r) + (1 - epsilon_d) * (z_d**r))**(1/r)
    
    c_agents = (alpha/(p_c*(alpha+beta+gamma))) * (phi*w*(1-tau_w)*t + l - p_d*d0)
    d_agents = (beta/(p_d*(alpha+beta+gamma))) * (phi*w*(1-tau_w)*t + l - p_d*d0) + d0
    l_agents = (gamma/((alpha+beta+gamma)*(1-tau_w)*phi*w)) * (phi*w*(1-tau_w)*t + l - p_d*d0)
    
    agg_c = np.sum(c_agents)
    agg_d = np.sum(d_agents)
    agg_labor = np.sum(phi*(t - l_agents))
    
    profit_c = f_c - w*t_c - tau_z*z_c
    profit_d = p_d*f_d - w*t_d - tau_z*z_d
    
    budget_errors = np.zeros(n)
    for i in range(n):
        income = phi[i]*w*(1-tau_w[i])*(t - l_agents[i]) + l
        expenditure = c_agents[i] + p_d*d_agents[i]
        budget_errors[i] = income - expenditure
    
    utilities = np.zeros(n)
    for i in range(n):
        if c_agents[i] > 0 and (d_agents[i]-d0) > 0 and l_agents[i] > 0:
            utilities[i] = alpha*np.log(c_agents[i]) + beta*np.log(d_agents[i]-d0) + gamma*np.log(l_agents[i])
        else:
            utilities[i] = -1e6  
    
    agg_utility = np.sum(utilities)
    
    results = {
        "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d, "w": w, "p_d": p_d, "l": l,
        "f_c": f_c, "f_d": f_d,
        "c_agents": c_agents, "d_agents": d_agents, "l_agents": l_agents,
        "agg_c": agg_c, "agg_d": agg_d, "agg_labor": agg_labor,
        "agg_utility": agg_utility,
        "profit_c": profit_c, "profit_d": profit_d,
        "budget_errors": budget_errors,
        "utilities": utilities,
        "sol": sol,
        "system_residuals": system_eqns(sol.x)  # include residual vector
    }
    
    # --- Output all residuals ---
    residuals = results["system_residuals"]
    print("\nResiduals from system equations:")
    for idx, res_val in enumerate(residuals, start=1):
        print(f"  eq{idx} residual: {res_val:.6e}")
    
    gov_budget_residual = residuals[6]
    print(f"\nGovernment budget condition (eq7) residual: {gov_budget_residual:.6e}")
    
    # For firm c: t_c/z_c = [((1-ε_c)*w)/(τ_zε_c)]^(1/(1-r))
    expected_ratio_c = (((1 - epsilon_c) * w) / (tau_z * epsilon_c)) ** (1/(r-1))
    computed_ratio_c = t_c / z_c
    residual_c = computed_ratio_c - expected_ratio_c
    print(f"\nFirm c condition:")
    print(f"  t_c/z_c = {computed_ratio_c:.6e}")
    print(f"  Expected: {expected_ratio_c:.6e}")
    print(f"  Residual: {residual_c:.6e}")
    
    # For firm d: t_d/z_d = [((1-ε_d)*w)/(τ_zε_d)]^(1/(1-r))
    expected_ratio_d = (((1 - epsilon_d) * w) / (tau_z * epsilon_d)) ** (1/(r-1))
    computed_ratio_d = t_d / z_d
    residual_d = computed_ratio_d - expected_ratio_d
    print(f"\nFirm d condition:")
    print(f"  t_d/z_d = {computed_ratio_d:.6e}")
    print(f"  Expected: {expected_ratio_d:.6e}")
    print(f"  Residual: {residual_d:.6e}")
    
    # --- Print aggregate pollution (z_c + z_d) ---
    agg_z = z_c + z_d
    print(f"\nAggregate z (pollution): {agg_z:.4f}")
    
    return sol.x, results, sol.success

# Example run
#tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
#tau_w = np.array([-1.12963781, -0.06584074,  0.2043803,   0.38336986,  0.63241591])
tau_z = 4.0
g = 5.0
    
solution, results, converged = solve(tau_w, tau_z, g)
    
print("\nsolution status:", results["sol"].status)
print("solution message:", results["sol"].message)
print("convergence:", converged)
print("solution vector [t_c, t_d, log_z_c, log_z_d, w, p_d, l]:")
print(solution)
    
print("\nproduction summary:")
print(f"  sector c: t_prod = {results['t_c']:.4f}, z_c = {results['z_c']:.4f}")
print(f"  sector d: t_prod = {results['t_d']:.4f}, z_d = {results['z_d']:.4f}")
    
print("\nhousehold demands and leisure:")
for i in range(n):
    print(f"  household {i+1}: c = {results['c_agents'][i]:.4f}, d = {results['d_agents'][i]:.4f}, l = {results['l_agents'][i]:.4f}")
    
print("\nAggregate results:")
print(f"  Aggregate c: {results['agg_c']:.4f}")
print(f"  Aggregate d: {results['agg_d']:.4f}")
print(f"  Aggregate leisure (labor): {results['agg_labor']:.4f}")
print(f"  Aggregate utility: {results['agg_utility']:.4f}")
    
agg_pollution = results["z_c"] + results["z_d"]
print(f"\nAggregate Pollution: {agg_pollution:.4f}")
    
print("\nhousehold utilities:")
for i in range(n):
    print(f"  household {i+1}: utility = {results['utilities'][i]:.4f}")
    