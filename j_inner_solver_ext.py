import numpy as np
from scipy.optimize import root

# a. baseline parameters
t = 24.0
d0 = 0.5
alpha = 0.7
beta = 0.2
gamma = 0.2
epsilon_c = 0.995  
epsilon_d = 0.92   
sigma = 0.5        
p_c = 1.0          
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.511])
varphi = np.array([0.03, 0.04, 0.08, 0.15, 0.70])
n = len(phi)
p_a = 5.0         
epsilon_z = 0.82  

# b. solve for general equilibrium
def solve(tau_w, tau_z, g, p_a, varsigma):

    # b.1 reformulate ces exponents for simplicity
    a = (sigma - 1.0) / sigma 
    b = (varsigma - 1.0) / varsigma
     
    # b.2 form system of equations
    def system_eqns(y):
        
        # unknowns vector
        t_c, t_d, lz_c, lz_d, la_c, la_d, w, p_d, l = y
        
        # transformations
        z_c, z_d = np.exp(lz_c), np.exp(lz_d)
        a_c, a_d = np.exp(la_c), np.exp(la_d)
        
        # check feasibility, penalize (return large value) if infeasible
        if min(t_c, t_d, z_c, z_d, a_c, a_d, w, p_d, tau_z) <= 0:
            return np.ones(9) * 1e6

        # define production
        base_c = epsilon_z * z_c**b + (1 - epsilon_z) * a_c**b
        base_d = epsilon_z * z_d**b + (1 - epsilon_z) * a_d**b
        if base_c <= 0 or base_d <= 0:
            return np.ones(9) * 1e6
        inner_c = base_c**(1/b)
        inner_d = base_d**(1/b)
        base_fc = epsilon_c * t_c**a + (1 - epsilon_c) * inner_c**a
        base_fd = epsilon_d * t_d**a + (1 - epsilon_d) * inner_d**a
        if base_fc <= 0 or base_fd <= 0:
            return np.ones(9) * 1e6
        f_c = base_fc**(1/a)
        f_d = base_fd**(1/a)

        # define firms' focs
        MPL_c = epsilon_c * (f_c / t_c)**(1 - a)
        MPL_d = epsilon_d * (f_d / t_d)**(1 - a)
        MPZ_c = (1 - epsilon_c) * epsilon_z * (f_c/inner_c)**(1 - a) * (inner_c/z_c)**(1 - b)
        MPZ_d = (1 - epsilon_d) * epsilon_z * (f_d/inner_d)**(1 - a) * (inner_d/z_d)**(1 - b)
        MPA_c = (1 - epsilon_c)*(1 - epsilon_z)*(f_c/inner_c)**(1 - a)*(inner_c/a_c)**(1 - b)
        MPA_d = (1 - epsilon_d)*(1 - epsilon_z)*(f_d/inner_d)**(1 - a)*(inner_d/a_d)**(1 - b)

        # define households' potential income
        income = phi*w*(1 - tau_w)*t + (1 - tau_w)*(a_d+a_c)*varphi*p_a + l - p_d*d0
        
        # penalize if income is negative
        if np.any(income <= 0):
            return np.ones(9) * 1e6
        
        # define households' dirty good demand
        d_i = (beta/(p_d*(alpha+beta+gamma))) * income + d0
        
        # define denominator for leisure definition, penalize if unfeasible
        denom = (alpha+beta+gamma)*(1 - tau_w)*phi*w
        if np.any(denom <= 0):
            return np.ones(9) * 1e6
        
        # compute households' leisure choice. penalize if infeasivle
        ell_i = (gamma/denom) * income
        if np.any(ell_i <= 0) or np.any(ell_i >= t):
            return np.ones(9) * 1e6

        # compute aggregate labor supply and aggregate dirty good demand
        agg_labor = np.sum(phi*(t - ell_i))
        agg_d     = np.sum(d_i)
        
        # define income tax and environmental tax revenue
        inc_rev = np.sum(tau_w*w*phi*(t - ell_i) + tau_w*(a_d+a_c)*varphi*p_a) 
        env_rev = tau_z * (z_c + z_d)
        
        # define equilibrium equations
        eq1 = t_c + t_d - agg_labor # labor market clearing
        eq2 = agg_d + 0.5*g/p_d - f_d # dirty good market clearing
        
        eq3 = p_c*MPL_c - w # firm c labor foc
        eq4 = p_c*MPZ_c - tau_z # firm c pollution foc
        
        eq5 = p_d*MPL_d - w # firm d labor foc
        eq6 = p_d*MPZ_d - tau_z # firm d pollution foc
        
        eq8 = p_c*MPA_c - p_a # firm c clean tech. foc
        eq9 = p_d*MPA_d - p_a # firm d clean tech. foc
        
        eq7 = n*l - (inc_rev + env_rev - g) # gov't budget constraint
        
        # return equation array
        return np.array([eq1,eq2,eq3,eq4,eq5,eq6,eq7,eq8,eq9])

    # c. solve system of equations for unknowns vector
    
    # c.1 initial guess
    y0 = np.array([10.0,10.0, np.log(1.0),np.log(1.0), np.log(0.5),np.log(0.5), 0.8,1.5,1.0])
    
    # c.2 solve using lm
    sol = root(system_eqns, y0, method='lm', options={'ftol':1e-20})
    
    # c.3 report non-convergence
    if not sol.success:
        return None, None, False

    # c.4 unpack solution
    t_c, t_d, lz_c, lz_d, la_c, la_d, w, p_d, l = sol.x
    z_c, z_d = np.exp(lz_c), np.exp(lz_d)
    a_c, a_d = np.exp(la_c), np.exp(la_d)

    # c.5 compute equilibrium production
    inner_c = (epsilon_z*z_c**b + (1 - epsilon_z)*a_c**b)**(1/b)
    inner_d = (epsilon_z*z_d**b + (1 - epsilon_z)*a_d**b)**(1/b)
    f_c = (epsilon_c*t_c**a + (1 - epsilon_c)*inner_c**a)**(1/a)
    f_d = (epsilon_d*t_d**a + (1 - epsilon_d)*inner_d**a)**(1/a)

    # c.6 compute equilibrium allocations for all households
    income = phi*w*(1 - tau_w)*t + (1 - tau_w)*(a_d+a_c)*varphi*p_a + l - p_d*d0
    d_i = (beta/(p_d*(alpha+beta+gamma))) * income + d0
    ell_i = (gamma/((alpha+beta+gamma)*(1 - tau_w)*phi*w)) * income

    # c.7 compute equilibrium aggregates
    agg_c    = np.sum((alpha/(p_c*(alpha+beta+gamma))) * income)
    agg_d    = np.sum(d_i)
    agg_lab  = np.sum(phi*(t - ell_i))

    # c.8 compute equilibrium profits
    profit_c = p_c*f_c - w*t_c - tau_z*z_c - p_a*a_c
    profit_d = p_d*f_d - w*t_d - tau_z*z_d - p_a*a_d

    # c.9 compute equilibrium log blue utilities for all households
    utilities = np.zeros(n)
    
    # c.10 compute equilibrium budget errors
    budget_errors = np.zeros(n)
    for i in range(n):
        c_i = (alpha/(p_c*(alpha+beta+gamma))) * income[i]
        utilities[i] = alpha*np.log(c_i) + beta*np.log(d_i[i] - d0) + gamma*np.log(ell_i[i])
        income_i = phi[i]*w*(1 - tau_w[i])*(t - ell_i[i]) + (1 - tau_w[i])*(a_d+a_c)*varphi[i]*p_a + l
        exp_i    = p_c*c_i + p_d*d_i[i]
        budget_errors[i] = income_i - exp_i

    # c.11 results
    results = {
        't_c':t_c, 't_d':t_d, 'z_c':z_c, 'z_d':z_d, 'a_c':a_c, 'a_d':a_d,
        'w':w, 'p_d':p_d, 'l':l,
        'f_c':f_c, 'f_d':f_d,
        'c_agents':(alpha/(p_c*(alpha+beta+gamma))) * income,
        'd_agents':d_i, 'ell_agents':ell_i,
        'agg_c':agg_c, 'agg_d':agg_d, 'agg_labor':agg_lab,
        'profit_c':profit_c, 'profit_d':profit_d,
        'utilities':utilities, 'budget_errors':budget_errors,
        'system_residuals':sol.fun
    }
    
    # c.12 return solution and results
    return sol.x, results, True

# d. test the solver

# d.1 test parameters
p_a_test = 5.0
varsig_test = 2.0
tau_w_test = np.array([0.015, 0.072, 0.115, 0.156, 0.24])

# d.2 solve for general equilibrium
sol_vec, res, conv = solve(tau_w_test, 1.0, 5.0, p_a_test, varsig_test)

# d.3 print results
print('converged:', conv)
if conv:
    print('solution:', sol_vec)
    for idx, r in enumerate(res['system_residuals'],1):
        print(f'eq{idx}: {r:.2e}')
    print('profits:', res['profit_c'], res['profit_d'])
