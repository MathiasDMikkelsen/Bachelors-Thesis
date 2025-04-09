import numpy as np
from scipy.optimize import root

# parameters (except φ now)
alpha = 0.7
beta = 0.2
gamma = 0.2
r = -1.0
t = 24.0     
d0 = 0.5
epsilon_c = 0.995     
epsilon_d = 0.92   
p_c = 1.0      

def solve(tau_w, tau_z, g, phi):
    n = len(phi)  # now using the passed φ
    
    def system_eqns(y):
        t_c, t_d, log_z_c, log_z_d, w, p_d, l = y
        z_c = np.exp(log_z_c)
        z_d = np.exp(log_z_d)
        
        f_c = (epsilon_c * (t_c**r) + (1 - epsilon_c) * (z_c**r))**(1/r)
        f_d = (epsilon_d * (t_d**r) + (1 - epsilon_d) * (z_d**r))**(1/r)
        
        d_agents = (beta/(p_d*(alpha+beta+gamma)))*(phi * w * (1-tau_w) * t + l - p_d*d0) + d0
        l_agents = (gamma/((alpha+beta+gamma)*(1-tau_w)*phi*w))*(phi * w * (1-tau_w) * t + l - p_d*d0)
        
        agg_labor = np.sum(phi * (t - l_agents))
        agg_d = np.sum(d_agents)
    
        eq1 = t_c + t_d - agg_labor
        eq2 = (agg_d + 0.5*g/p_d) - f_d
        eq3 = w - epsilon_c * (t_c**(r-1)) * (f_c**(1-r))
        eq4 = tau_z - (1 - epsilon_c) * (z_c**(r-1)) * (f_c**(1-r))
        eq5 = w - epsilon_d * (t_d**(r-1)) * (f_d**(1-r)) * p_d
        eq6 = tau_z - (1 - epsilon_d) * (z_d**(r-1)) * (f_d**(1-r)) * p_d
        eq7 = n*l - (np.sum(tau_w * w * phi * (t - l_agents)) + tau_z*(z_c+z_d) - g)
    
        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7])
    
    y0 = np.array([0.3, 0.4, np.log(0.6), np.log(0.4), 0.5, 1.5, 0.1])
    
    sol = root(system_eqns, y0, method='lm')
    t_c, t_d, log_z_c, log_z_d, w, p_d, l = sol.x
    z_c = np.exp(log_z_c)
    z_d = np.exp(log_z_d)
    
    f_c = (epsilon_c * (t_c**r) + (1 - epsilon_c) * (z_c**r))**(1/r)
    f_d = (epsilon_d * (t_d**r) + (1 - epsilon_d) * (z_d**r))**(1/r)
    
    c_agents = (alpha/(p_c*(alpha+beta+gamma)))*(phi * w * (1-tau_w) * t + l - p_d*d0)
    d_agents = (beta/(p_d*(alpha+beta+gamma)))*(phi * w * (1-tau_w) * t + l - p_d*d0) + d0
    l_agents = (gamma/((alpha+beta+gamma)*(1-tau_w)*phi*w))*(phi * w * (1-tau_w) * t + l - p_d*d0)
    
    budget_errors = np.array([phi[i]*w*(1-tau_w[i])*(t-l_agents[i]) + l - (c_agents[i] + p_d*d_agents[i]) for i in range(n)])
    
    utilities = np.zeros(n)
    for i in range(n):
        if c_agents[i] > 0 and (d_agents[i]-d0) > 0 and l_agents[i] > 0:
            utilities[i] = alpha*np.log(c_agents[i]) + beta*np.log(d_agents[i]-d0) + gamma*np.log(l_agents[i])
        else:
            utilities[i] = -1e6
    
    results = {
        "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d, "w": w, "p_d": p_d, "l": l,
        "c_agents": c_agents, "d_agents": d_agents, "l_agents": l_agents,
        "budget_errors": budget_errors,
        "utilities": utilities,
        "sol": sol
    }
    
    return sol.x, results, sol.success