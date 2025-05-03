import numpy as np
from scipy.optimize import root

# a. Baseline parameters
t = 24.0
d0 = 0.5
alpha = 0.7
beta = 0.2
gamma = 0.2
epsilon_c = 0.995  # Labor share parameter for clean firm
epsilon_d = 0.92   # Labor share parameter for dirty firm
sigma = 0.5        # Elasticity sigma (labor vs composite)
p_c = 1.0          # Numeraire
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.511])
varphi = np.array([0.03, 0.04, 0.08, 0.15, 0.70])
n = len(phi)

# b. Abatement extension parameters
p_a = 5.0         # Abatement price
epsilon_z = 0.82  # Z/A nest share
def solve(tau_w, tau_z, g, p_a, varsigma):
    """
    Solve GE with abatement-nest elasticity `varsigma`.
    tau_w: array of income tax rates (length n)
    tau_z: scalar environmental tax
    g:     government spending
    p_a:   abatement price
    varsigma: elasticity in Z/A nest
    Returns: sol.x, results dict, success flag
    """
    # CES exponents
    a = (sigma - 1.0) / sigma if sigma != 1.0 else -np.inf
    b = (varsigma - 1.0) / varsigma if varsigma != 1.0 else -np.inf

    assert len(tau_w) == n, f"tau_w must have length n={n}"
    assert 0 < epsilon_z < 1, "epsilon_z must be between 0 and 1"

    def system_eqns(y):
        t_c, t_d, lz_c, lz_d, la_c, la_d, w, p_d, l = y
        # exponentiate logs
        z_c, z_d = np.exp(lz_c), np.exp(lz_d)
        a_c, a_d = np.exp(la_c), np.exp(la_d)
        # feasibility
        if min(t_c, t_d, z_c, z_d, a_c, a_d, w, p_d, tau_z) <= 0:
            return np.ones(9) * 1e6

        # inner Z/A composite
        if np.isinf(b):
            inner_c = z_c**epsilon_z * a_c**(1 - epsilon_z)
            inner_d = z_d**epsilon_z * a_d**(1 - epsilon_z)
        else:
            base_c = epsilon_z * z_c**b + (1 - epsilon_z) * a_c**b
            base_d = epsilon_z * z_d**b + (1 - epsilon_z) * a_d**b
            if base_c <= 0 or base_d <= 0:
                return np.ones(9) * 1e6
            inner_c = base_c**(1/b)
            inner_d = base_d**(1/b)

        # top-level CES
        if np.isinf(a):
            f_c = t_c**epsilon_c * inner_c**(1 - epsilon_c)
            f_d = t_d**epsilon_d * inner_d**(1 - epsilon_d)
        else:
            base_fc = epsilon_c * t_c**a + (1 - epsilon_c) * inner_c**a
            base_fd = epsilon_d * t_d**a + (1 - epsilon_d) * inner_d**a
            if base_fc <= 0 or base_fd <= 0:
                return np.ones(9) * 1e6
            f_c = base_fc**(1/a)
            f_d = base_fd**(1/a)

        # firm FOCs
        MPL_c = epsilon_c * (f_c / t_c)**(1 - a)
        MPL_d = epsilon_d * (f_d / t_d)**(1 - a)
        MPZ_c = (1 - epsilon_c) * epsilon_z * (f_c/inner_c)**(1 - a) * (inner_c/z_c)**(1 - b)
        MPZ_d = (1 - epsilon_d) * epsilon_z * (f_d/inner_d)**(1 - a) * (inner_d/z_d)**(1 - b)
        MPA_c = (1 - epsilon_c)*(1 - epsilon_z)*(f_c/inner_c)**(1 - a)*(inner_c/a_c)**(1 - b)
        MPA_d = (1 - epsilon_d)*(1 - epsilon_z)*(f_d/inner_d)**(1 - a)*(inner_d/a_d)**(1 - b)

        # household behavior
        income = phi*w*(1 - tau_w)*t + (1 - tau_w)*(a_d+a_c)*varphi*p_a + l - p_d*d0
        if np.any(income <= 0):
            return np.ones(9) * 1e6
        d_i = (beta/(p_d*(alpha+beta+gamma))) * income + d0
        denom = (alpha+beta+gamma)*(1 - tau_w)*phi*w
        if np.any(denom <= 0):
            return np.ones(9) * 1e6
        l_i = (gamma/denom) * income
        if np.any(l_i <= 0) or np.any(l_i >= t):
            return np.ones(9) * 1e6

        agg_labor = np.sum(phi*(t - l_i))
        agg_d     = np.sum(d_i)

        # equilibrium equations
        eq1 = t_c + t_d - agg_labor
        eq2 = agg_d + 0.5*g/p_d - f_d
        eq3 = p_c*MPL_c - w
        eq4 = p_c*MPZ_c - tau_z
        eq5 = p_d*MPL_d - w
        eq6 = p_d*MPZ_d - tau_z
        inc_rev = np.sum(tau_w*w*phi*(t - l_i) + tau_w*(a_d+a_c)*varphi*p_a)
        env_rev = tau_z * (z_c + z_d)
        eq7 = n*l - (inc_rev + env_rev - g)
        eq8 = p_c*MPA_c - p_a
        eq9 = p_d*MPA_d - p_a
        return np.array([eq1,eq2,eq3,eq4,eq5,eq6,eq7,eq8,eq9])

    # initial guess
    y0 = np.array([10.0,10.0, np.log(1.0),np.log(1.0), np.log(0.5),np.log(0.5), 0.8,1.5,1.0])
    sol = root(system_eqns, y0, method='lm', options={'ftol':1e-20})
    if not sol.success:
        return None, None, False

    # unpack
    t_c, t_d, lz_c, lz_d, la_c, la_d, w, p_d, l = sol.x
    z_c, z_d = np.exp(lz_c), np.exp(lz_d)
    a_c, a_d = np.exp(la_c), np.exp(la_d)

    # recompute composites & outputs
    if np.isinf(b):
        inner_c = z_c**epsilon_z * a_c**(1 - epsilon_z)
        inner_d = z_d**epsilon_z * a_d**(1 - epsilon_z)
    else:
        inner_c = (epsilon_z*z_c**b + (1 - epsilon_z)*a_c**b)**(1/b)
        inner_d = (epsilon_z*z_d**b + (1 - epsilon_z)*a_d**b)**(1/b)
    if np.isinf(a):
        f_c = t_c**epsilon_c * inner_c**(1 - epsilon_c)
        f_d = t_d**epsilon_d * inner_d**(1 - epsilon_d)
    else:
        f_c = (epsilon_c*t_c**a + (1 - epsilon_c)*inner_c**a)**(1/a)
        f_d = (epsilon_d*t_d**a + (1 - epsilon_d)*inner_d**a)**(1/a)

    # households for results
    income = phi*w*(1 - tau_w)*t + (1 - tau_w)*(a_d+a_c)*varphi*p_a + l - p_d*d0
    d_i = (beta/(p_d*(alpha+beta+gamma))) * income + d0
    l_i = (gamma/((alpha+beta+gamma)*(1 - tau_w)*phi*w)) * income

    agg_c    = np.sum((alpha/(p_c*(alpha+beta+gamma))) * income)
    agg_d    = np.sum(d_i)
    agg_lab  = np.sum(phi*(t - l_i))

    profit_c = p_c*f_c - w*t_c - tau_z*z_c - p_a*a_c
    profit_d = p_d*f_d - w*t_d - tau_z*z_d - p_a*a_d

    utilities = np.zeros(n)
    budget_errors = np.zeros(n)
    for i in range(n):
        c_i = (alpha/(p_c*(alpha+beta+gamma))) * income[i]
        utilities[i] = alpha*np.log(c_i) + beta*np.log(d_i[i] - d0) + gamma*np.log(l_i[i])
        income_i = phi[i]*w*(1 - tau_w[i])*(t - l_i[i]) + (1 - tau_w[i])*(a_d+a_c)*varphi[i]*p_a + l
        exp_i    = p_c*c_i + p_d*d_i[i]
        budget_errors[i] = income_i - exp_i

    results = {
        't_c':t_c, 't_d':t_d, 'z_c':z_c, 'z_d':z_d, 'a_c':a_c, 'a_d':a_d,
        'w':w, 'p_d':p_d, 'l':l,
        'f_c':f_c, 'f_d':f_d,
        'c_agents':(alpha/(p_c*(alpha+beta+gamma))) * income,
        'd_agents':d_i, 'l_agents':l_i,
        'agg_c':agg_c, 'agg_d':agg_d, 'agg_labor':agg_lab,
        'profit_c':profit_c, 'profit_d':profit_d,
        'utilities':utilities, 'budget_errors':budget_errors,
        'system_residuals':sol.fun
    }
    return sol.x, results, True

# Example run for testing
if __name__ == '__main__':
    p_a_test = 5.0
    varsig_test = 2.0
    tau_w_test = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
    sol_vec, res, conv = solve(tau_w_test, 1.0, 5.0, p_a_test, varsig_test)
    print('converged:', conv)
    if conv:
        print('solution:', sol_vec)
        for idx, r in enumerate(res['system_residuals'],1):
            print(f'eq{idx}: {r:.2e}')
        print('profits:', res['profit_c'], res['profit_d'])
