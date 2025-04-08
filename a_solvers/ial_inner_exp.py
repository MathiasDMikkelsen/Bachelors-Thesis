import numpy as np
from scipy.optimize import root

# a. model parameters
alpha = 0.24
beta = 0.45
gamma = 0.31
d0 = 0.2
phi = np.array([0.6, 0.8, 1.0, 1.2, 1.4])
n = phi.size

# b. production parameters
theta_c = 0.33
theta_d = 0.66
varphi = 0.5
epsilon = 0.3
T = 24  # Maximum labor supply

def solve(tau_w, tau_z, G):
    """Solve the general equilibrium model."""

    # 1. set up
    # ensure tau_w is an array
    tau_w = np.array(tau_w)

    # a. exogenous prices
    T_C = 1.0
    T_D = 1.0
    Z_C = 1.0
    Z_D = 1.0
    w = 1.0
    p_D = 1.0
    L = 1.0

    # b. objective function
    def obj(x):
        """Objective function to find the root of."""

        # i. rename
        T_C = x[0]
        T_D = x[1]
        Z_C = x[2]
        Z_D = x[3]
        w = x[4]
        p_D = x[5]
        L = x[6]

        # ii. implied values

        # a. firm behavior
        K_C = (theta_c * w / T_C)**(1/(1-theta_c)) * L
        K_D = (theta_d * w / (p_D * T_D))**(1/(1-theta_d)) * L
        f_C = T_C * K_C**(theta_c) * L**(1-theta_c)
        f_D = T_D * K_D**(theta_d) * L**(1-theta_d)
        profit_c = f_C - w*L - T_C*K_C
        profit_d = p_D*f_D - w*L - T_D*K_D

        # b. household behavior
        budget_errors = np.zeros(n)
        c_agents = np.zeros(n)
        d_agents = np.zeros(n)
        l_agents = np.zeros(n)
        u_agents = np.zeros(n)

        agg_c = 0.0
        agg_d = 0.0
        agg_labor = 0.0

        for i in range(n):
            # 1. labor supply
            l_supply = (gamma / (1-tau_w[i])) * (1 / w) * (alpha * w * (1-tau_w[i]) / 1 + beta * p_D) * (phi[i] * w * T)
            # 2. demands
            c = alpha * (phi[i] * w * (T - l_supply) + L) / 1
            d = beta * (phi[i] * w * (T - l_supply) + L) / p_D + d0

            # 3. utility
            u = alpha * np.log(c) + beta * np.log(d - d0) + gamma * np.log(l_supply)

            # 4. budget error
            budget_error = (phi[i] * w * (T - l_supply) + L) - (c + p_D * (d - d0))

            # 5. store
            c_agents[i] = c
            d_agents[i] = d
            l_agents[i] = l_supply
            u_agents[i] = u
            budget_errors[i] = budget_error

            # 6. aggregate
            agg_c += c
            agg_d += d
            agg_labor += (T - l_supply)

        # c. market clearing
        good_c_mkt_clearing = (agg_c + varphi*G) - f_C
        good_d_mkt_clearing = agg_d - f_D
        labor_mkt_clearing = agg_labor - L
        pollution_mkt_clearing = (Z_C + Z_D) - G

        # iii. return
        return np.array([
            good_c_mkt_clearing,
            good_d_mkt_clearing,
            labor_mkt_clearing,
            pollution_mkt_clearing,
            profit_c + tau_z* (Z_C + Z_D),
            profit_d,
            np.sum(tau_w*phi*w*(T-l_agents)) + tau_z*(Z_C + Z_D) - L
        ])

    # c. constraints
    def constraint_c(x):
        """Consumption must be positive."""
        c_agents = np.zeros(n)
        L = x[6]
        w = x[4]
        p_D = x[5]
        for i in range(n):
            l_supply = (gamma / (1-tau_w[i])) * (1 / w) * (alpha * w * (1-tau_w[i]) / 1 + beta * p_D) * (phi[i] * w * T)
            c = alpha * (phi[i] * w * (T - l_supply) + L) / 1
            c_agents[i] = c
        return np.min(c_agents)

    def constraint_d(x):
        """Demand must be greater than d0."""
        d_agents = np.zeros(n)
        L = x[6]
        w = x[4]
        p_D = x[5]
        for i in range(n):
            l_supply = (gamma / (1-tau_w[i])) * (1 / w) * (alpha * w * (1-tau_w[i]) / 1 + beta * p_D) * (phi[i] * w * T)
            d = beta * (phi[i] * w * (T - l_supply) + L) / p_D + d0
            d_agents[i] = d
        return np.min(d_agents) - d0

    def constraint_ell(x):
        """Leisure must be positive."""
        l_agents = np.zeros(n)
        L = x[6]
        w = x[4]
        p_D = x[5]
        for i in range(n):
            l_supply = (gamma / (1-tau_w[i])) * (1 / w) * (alpha * w * (1-tau_w[i]) / 1 + beta * p_D) * (phi[i] * w * T)
            l_agents[i] = l_supply
        return np.min(l_agents)

    def constraint_T(x):
        """Total labor supply must be positive."""
        L = x[6]
        return L

    # d. initial guess
    initial_guess = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    # e. find root
    constraints = ({'type': 'ineq', 'fun': constraint_c},
                   {'type': 'ineq', 'fun': constraint_d},
                   {'type': 'ineq', 'fun': constraint_ell},
                   {'type': 'ineq', 'fun': constraint_T})

    sol = root(obj, initial_guess, method='lm', constraints=constraints)

    # f. check convergence
    converged = sol.success
    message = sol.message
    x = sol.x

    # g. report
    if not converged:
        print(f'GE solve failed: {message}')
        return None, None, False

    # h. implied values
    T_C = x[0]
    T_D = x[1]
    Z_C = x[2]
    Z_D = x[3]
    w = x[4]
    p_D = x[5]
    L = x[6]

    K_C = (theta_c * w / T_C)**(1/(1-theta_c)) * L
    K_D = (theta_d * w / (p_D * T_D))**(1/(1-theta_d)) * L
    f_C = T_C * K_C**(theta_c) * L**(1-theta_c)
    f_D = T_D * K_D**(theta_d) * L**(1-theta_d)
    profit_c = f_C - w*L - T_C*K_C
    profit_d = p_D*f_D - w*L - T_D*K_D

    budget_errors = np.zeros(n)
    c_agents = np.zeros(n)
    d_agents = np.zeros(n)
    l_agents = np.zeros(n)
    utilities = np.zeros(n)

    agg_c = 0.0
    agg_d = 0.0
    agg_labor = 0.0

    for i in range(n):
        # 1. labor supply
        l_supply = (gamma / (1-tau_w[i])) * (1 / w) * (alpha * w * (1-tau_w[i]) / 1 + beta * p_D) * (phi[i] * w * T)

        # 2. demands
        c = alpha * (phi[i] * w * (T - l_supply) + L) / 1
        d = beta * (phi[i] * w * (T - l_supply) + L) / p_D + d0

        # 3. utility
        u = alpha * np.log(c) + beta * np.log(d - d0) + gamma * np.log(l_supply)

        # 4. budget error
        budget_error = (phi[i] * w * (T - l_supply) + L) - (c + p_D * (d - d0))

        # 5. store
        c_agents[i] = c
        d_agents[i] = d
        l_agents[i] = l_supply
        utilities[i] = u
        budget_errors[i] = budget_error

        # 6. aggregate
        agg_c += c
        agg_d += d
        agg_labor += (T - l_supply)

    # i. return
    results = {
        'T_C': T_C, 'T_D': T_D, 'Z_C': Z_C, 'Z_D': Z_D, 'w': w, 'p_D': p_D, 'L': L,
        'K_C': K_C, 'K_D': K_D, 'f_C': f_C, 'f_D': f_D, 'profit_c': profit_c, 'profit_d': profit_d,
        'c_agents': c_agents, 'd_agents': d_agents, 'l_agents': l_agents, 'utilities': utilities, 'budget_errors': budget_errors,
        'agg_c': agg_c, 'agg_d': agg_d, 'agg_labor': agg_labor,
        'z_c': Z_C, 'z_d': Z_D, 't_c': T_C, 't_d': T_D,
        'sol': sol
    }
    return x, results, converged