import numpy as np
from scipy.optimize import root

# Global model parameters
alpha     = 0.7
beta      = 0.2
gamma     = 0.2
r         = -1.0
T_val     = 24.0      # Time endowment per household
D0        = 0.5
epsilon_C = 0.995     # Example value for Sector C
epsilon_D = 0.92      # Example value for Sector D
p_C       = 1.0       # Numeraire
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n = len(phi)

def solve(tau_w, tau_z, G):
    """
    Solves the system of equations for the given tax rates and government spending.
    In this version, we reparameterize Z_C and Z_D to ensure they are always positive
    by solving for log_Z_C and log_Z_D.
    
    The decision vector for the system becomes:
       [T_C, T_D, log_Z_C, log_Z_D, w, p_D, L]
    """
    def system_eqns(y):
        # Unpack reparameterized variables:
        T_C, T_D, log_Z_C, log_Z_D, w, p_D, L = y
        Z_C = np.exp(log_Z_C)
        Z_D = np.exp(log_Z_D)
        
        # Calculate firm productions using CES functions:
        F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
        F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)
        
        # Household demands and leisure using standard formulas:
        D_agents = (beta/(p_D*(alpha+beta+gamma)))*(phi*w*(1-tau_w)*T_val + L - p_D*D0) + D0
        l_agents = (gamma/((alpha+beta+gamma)*(1-tau_w)*phi*w))*(phi*w*(1-tau_w)*T_val + L - p_D*D0)
        
        agg_labor = np.sum(phi*(T_val - l_agents))
        agg_D = np.sum(D_agents)
    
        eq1 = T_C + T_D - agg_labor
        eq2 = (agg_D + 0.5*G/p_D) - F_D
        eq3 = w - epsilon_C * (T_C**(r-1)) * (F_C**(1-r))
        eq4 = tau_z - (1 - epsilon_C) * (Z_C**(r-1)) * (F_C**(1-r))
        eq5 = w - epsilon_D * (T_D**(r-1)) * (F_D**(1-r)) * p_D
        eq6 = tau_z - (1 - epsilon_D) * (Z_D**(r-1)) * (F_D**(1-r)) * p_D
        eq7 = n*L - (np.sum(tau_w * w * phi * (T_val-l_agents)) + tau_z*(Z_C+Z_D) - G)
    
        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7])
    
    # Provide an initial guess. Note we use np.log for the Z variables.
    y0 = np.array([0.3, 0.4, np.log(0.6), np.log(0.4), 0.5, 1.5, 0.1])
    
    sol = root(system_eqns, y0, method='lm')
    
    # Unpack the solution and convert the log-values:
    T_C, T_D, log_Z_C, log_Z_D, w, p_D, L = sol.x
    Z_C = np.exp(log_Z_C)
    Z_D = np.exp(log_Z_D)
    
    # Compute additional results
    F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
    F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)
    
    C_agents = (alpha/(p_C*(alpha+beta+gamma)))*(phi*w*(1-tau_w)*T_val + L - p_D*D0)
    D_agents = (beta/(p_D*(alpha+beta+gamma)))*(phi*w*(1-tau_w)*T_val + L - p_D*D0) + D0
    l_agents = (gamma/((alpha+beta+gamma)*(1-tau_w)*phi*w))*(phi*w*(1-tau_w)*T_val + L - p_D*D0)
    
    agg_C = np.sum(C_agents)
    agg_D = np.sum(D_agents)
    agg_labor = np.sum(phi*(T_val - l_agents))
    
    profit_C = F_C - w*T_C - tau_z*Z_C
    profit_D = p_D*F_D - w*T_D - tau_z*Z_D
    
    budget_errors = np.zeros(n)
    for i in range(n):
        income = phi[i]*w*(1-tau_w[i])*(T_val - l_agents[i]) + L
        expenditure = C_agents[i] + p_D*D_agents[i]
        budget_errors[i] = income - expenditure
    
    utilities = np.zeros(n)
    for i in range(n):
        if C_agents[i] > 0 and (D_agents[i]-D0) > 0 and l_agents[i] > 0:
            utilities[i] = alpha*np.log(C_agents[i]) + beta*np.log(D_agents[i]-D0) + gamma*np.log(l_agents[i])
        else:
            utilities[i] = -1e6  # Heavy penalty if any component is non-positive
    
    results = {
        "T_C": T_C, "T_D": T_D, "Z_C": Z_C, "Z_D": Z_D, "w": w, "p_D": p_D, "L": L,
        "F_C": F_C, "F_D": F_D,
        "C_agents": C_agents, "D_agents": D_agents, "l_agents": l_agents,
        "agg_C": agg_C, "agg_D": agg_D, "agg_labor": agg_labor,
        "profit_C": profit_C, "profit_D": profit_D,
        "budget_errors": budget_errors,
        "utilities": utilities,
        "sol": sol
    }
    
    return sol.x, results, sol.success

# Example usage:
if __name__ == "__main__":
    tau_w = np.array([-1.75, -0.5, 0.0, 0.2, 0.6])
    tau_z = 1.25
    G = 5.0
    
    solution, results, converged = solve(tau_w, tau_z, G)
    
    print("Solution status:", results["sol"].status)
    print("Solution message:", results["sol"].message)
    print("Convergence:", converged)
    print("Solution vector [T_C, T_D, log_Z_C, log_Z_D, w, p_D, L]:")
    print(solution)
    
    print("\nProduction Summary:")
    print(f"Sector C: T_prod = {results['T_C']:.4f}, Z_C = {results['Z_C']:.4f}")
    print(f"Sector D: T_prod = {results['T_D']:.4f}, Z_D = {results['Z_D']:.4f}")
    
    print("\nHousehold Demands and Leisure:")
    for i in range(n):
        print(f"Household {i+1}: C = {results['C_agents'][i]:.4f}, D = {results['D_agents'][i]:.4f}, l = {results['l_agents'][i]:.4f}")
    
    print("\nHousehold Utilities:")
    for i in range(n):
        print(f"Household {i+1}: Utility = {results['utilities'][i]:.4f}")