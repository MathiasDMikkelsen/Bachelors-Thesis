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

# Productivity weights split by sector
phi_d = np.array([0.15*2, 0.35*2])   # polluting sector (sum = 0.5)
phi_c = np.array([0.1*2, 0.2*2, 0.2*2])  # clean sector (sum = 0.5)
phi = np.concatenate([phi_d, phi_c])  # total phi for all 5 households
n = len(phi)

def solve(tau_w, tau_z, g):

    def system_eqns(y):
        t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = y
        z_c = np.exp(log_z_c)
        z_d = np.exp(log_z_d)
        
        # Production functions
        f_c = (epsilon_c * (t_c**r) + (1 - epsilon_c) * (z_c**r))**(1/r)
        f_d = (epsilon_d * (t_d**r) + (1 - epsilon_d) * (z_d**r))**(1/r)

        # Wages per household (by sector)
        wage = np.concatenate([w_d * np.ones(2), w_c * np.ones(3)])

        # Income per household
        income = phi * (1 - tau_w) * wage * (t - l)
        total_income = income + l

        # Household decisions
        c_agents = (alpha / (p_c * (alpha + beta + gamma))) * total_income
        d_agents = (beta / (p_d * (alpha + beta + gamma))) * total_income + d0
        l_agents = (gamma / ((alpha + beta + gamma) * (1 - tau_w) * phi * wage)) * total_income

        # Aggregate labor inputs by sector
        agg_labor_d = np.sum(phi_d * (t - l_agents[:2]))
        agg_labor_c = np.sum(phi_c * (t - l_agents[2:]))

        # Aggregate demand for dirty good
        agg_d = np.sum(d_agents)

        # System of equations
        eq1 = t_c - agg_labor_c
        eq2 = t_d - agg_labor_d
        eq3 = (agg_d + 0.5 * g / p_d) - f_d
        eq4 = w_c - epsilon_c * (t_c**(r - 1)) * (f_c**(1 - r))
        eq5 = tau_z - (1 - epsilon_c) * (z_c**(r - 1)) * (f_c**(1 - r))
        eq6 = w_d - epsilon_d * (t_d**(r - 1)) * (f_d**(1 - r)) * p_d
        eq7 = tau_z - (1 - epsilon_d) * (z_d**(r - 1)) * (f_d**(1 - r)) * p_d
        eq8 = n * l - (np.sum(tau_w * phi * wage * (t - l_agents)) + tau_z * (z_c + z_d) - g)

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8])

    # Initial guess: [t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l]
    y0 = np.array([5.0, 5.0, np.log(0.6), np.log(0.4), 0.5, 0.6, 1.5, 0.1])

    sol = root(system_eqns, y0, method='lm')

    t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l = sol.x
    z_c = np.exp(log_z_c)
    z_d = np.exp(log_z_d)

    # Recalculate variables
    f_c = (epsilon_c * (t_c**r) + (1 - epsilon_c) * (z_c**r))**(1/r)
    f_d = (epsilon_d * (t_d**r) + (1 - epsilon_d) * (z_d**r))**(1/r)

    wage = np.concatenate([w_d * np.ones(2), w_c * np.ones(3)])
    income = phi * (1 - tau_w) * wage * (t - l)
    total_income = income + l

    c_agents = (alpha / (p_c * (alpha + beta + gamma))) * total_income
    d_agents = (beta / (p_d * (alpha + beta + gamma))) * total_income + d0
    l_agents = (gamma / ((alpha + beta + gamma) * (1 - tau_w) * phi * wage)) * total_income

    agg_c = np.sum(c_agents)
    agg_d = np.sum(d_agents)
    agg_labor = np.sum(phi * (t - l_agents))

    profit_c = f_c - w_c * t_c - tau_z * z_c
    profit_d = p_d * f_d - w_d * t_d - tau_z * z_d

    budget_errors = np.zeros(n)
    for i in range(n):
        income_i = phi[i] * wage[i] * (1 - tau_w[i]) * (t - l_agents[i]) + l
        expenditure_i = c_agents[i] + p_d * d_agents[i]
        budget_errors[i] = income_i - expenditure_i

    utilities = np.zeros(n)
    for i in range(n):
        if c_agents[i] > 0 and (d_agents[i] - d0) > 0 and l_agents[i] > 0:
            utilities[i] = alpha * np.log(c_agents[i]) + beta * np.log(d_agents[i] - d0) + gamma * np.log(l_agents[i])
        else:
            utilities[i] = -1e6

    residuals = system_eqns(sol.x)

    results = {
        "t_c": t_c, "t_d": t_d, "z_c": z_c, "z_d": z_d, "w_c": w_c, "w_d": w_d, "p_d": p_d, "l": l,
        "f_c": f_c, "f_d": f_d,
        "c_agents": c_agents, "d_agents": d_agents, "l_agents": l_agents,
        "agg_c": agg_c, "agg_d": agg_d, "agg_labor": agg_labor,
        "profit_c": profit_c, "profit_d": profit_d,
        "budget_errors": budget_errors,
        "utilities": utilities,
        "residuals": residuals,
        "sol": sol
    }

    return sol.x, results, sol.success

# Inputs
tau_w = np.array([-1.75, -0.5, 0.0, 0.25, 0.55])  # Klenert optimality
tau_z = 1.0
g = 5.0

# Solve model
solution, results, converged = solve(tau_w, tau_z, g)

# Output
print("solution status:", results["sol"].status)
print("solution message:", results["sol"].message)
print("convergence:", converged)
print("solution vector [t_c, t_d, log_z_c, log_z_d, w_c, w_d, p_d, l]:")
print(solution)

print("\nproduction summary:")
print(f"sector c: t_prod = {results['t_c']:.4f}, z_c = {results['z_c']:.4f}")
print(f"sector d: t_prod = {results['t_d']:.4f}, z_d = {results['z_d']:.4f}")

print("\nhousehold demands and leisure:")
for i in range(n):
    print(f"household {i+1}: c = {results['c_agents'][i]:.4f}, d = {results['d_agents'][i]:.4f}, l = {results['l_agents'][i]:.4f}")

print("\nhousehold utilities:")
for i in range(n):
    print(f"household {i+1}: utility = {results['utilities'][i]:.4f}")

print("\nresiduals of equations:")
for i, res in enumerate(results["residuals"], 1):
    print(f"Equation {i}: residual = {res:.4e}")
