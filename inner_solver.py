import numpy as np
from scipy.optimize import root

# Global model parameters
alpha     = 0.7
beta      = 0.2
gamma     = 0.2
r         = -1.0
T_val     = 24.0      # Time endowment per household
D0        = 0.5
epsilon_C = 0.995    # Example value for Sector C
epsilon_D = 0.92     # Example value for Sector D
p_C       = 1.0      # Numeraire
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175])
n = len(phi)

def solve(tau_w, tau_z, G):
    """
    Solves the system of equations for the given tax rates and government spending.

    Args:
        tau_w (np.ndarray): Array of labor tax rates for each household.
        tau_z (float): Tax rate on pollution.
        G (float): Government spending.

    Returns:
        tuple: A tuple containing the solution vector, a dictionary of results, and a boolean indicating convergence.
    """

    def system_eqns(x):
        T_C, T_D, Z_C, Z_D, w, p_D, L = x

        F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
        F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)
        
        D_agents = (beta/(p_D*(alpha+beta+gamma)))*(phi*w*(1-tau_w)*T_val + L - p_D*D0) + D0
        l_agents = (gamma/((alpha+beta+gamma)*(1-tau_w)*phi*w))*(phi*w*(1-tau_w)*T_val + L - p_D*D0)
        
        agg_labor = np.sum(phi*(T_val - l_agents))  
        agg_D = np.sum(D_agents)                     

        eq1 = T_C + T_D - agg_labor
        eq2 = (agg_D+0.5*G/p_D) - F_D
        eq3 = w - epsilon_C * (T_C**(r-1)) * (F_C**(1-r)) 
        eq4 = tau_z - (1 - epsilon_C) * (Z_C**(r-1)) * (F_C**(1-r))
        eq5 = w - epsilon_D * (T_D**(r-1)) * (F_D**(1-r)) * p_D
        eq6 = tau_z - (1 - epsilon_D) * (Z_D**(r-1)) * (F_D**(1-r)) * p_D
        eq7 = (n*L-(np.sum(tau_w*w*phi*(T_val-l_agents))+tau_z*(Z_C+Z_D)-G))

        return np.array([eq1, eq2, eq3, eq4, eq5, eq6, eq7])

    x0 = np.array([0.3, 0.4, 0.6, 0.4, 0.5, 1.5, 0.1])

    sol = root(system_eqns, x0, method='lm')

    # Extract solution values
    T_C, T_D, Z_C, Z_D, w, p_D, L = sol.x

    # Calculate additional results
    F_C = (epsilon_C * (T_C**r) + (1 - epsilon_C) * (Z_C**r))**(1/r)
    F_D = (epsilon_D * (T_D**r) + (1 - epsilon_D) * (Z_D**r))**(1/r)

    C_agents = (alpha/(p_C*(alpha+beta+gamma)))*(phi*w*(1-tau_w)*T_val + L - p_D*D0)
    D_agents = (beta/(p_D*(alpha+beta+gamma)))*(phi*w*(1-tau_w)*T_val + L - p_D*D0) + D0
    l_agents = (gamma/((alpha+beta+gamma)*(1-tau_w)*phi*w))*(phi*w*(1-tau_w)*T_val + L - p_D*D0)

    agg_C = np.sum(C_agents)
    agg_D = np.sum(D_agents)
    agg_labor = np.sum(phi*(T_val - l_agents))

    profit_C = F_C - w * T_C - tau_z * Z_C
    profit_D = p_D * F_D - w * T_D - tau_z * Z_D

    # Calculate budget constraint errors
    budget_errors = np.zeros(n)
    for i in range(n):
        income = phi[i]*w*(1-tau_w[i])*(T_val-l_agents[i]) + L
        expenditure = C_agents[i] + p_D*D_agents[i]
        budget_errors[i] = income - expenditure
        
    # Compute household utilities
    utilities = np.zeros(n)
    for i in range(n):
        if C_agents[i] > 0 and (D_agents[i] - D0) > 0 and l_agents[i] > 0:
            utilities[i] = alpha * np.log(C_agents[i]) + beta * np.log(D_agents[i] - D0) + gamma * np.log(l_agents[i])
        else:
            utilities[i] = 0  # Assign negative infinity if inputs are not positive

    # Store results in a dictionary
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
    tau_w = np.array([0.015, 0.072, 0.115, 0.156, 0.24])
    tau_z = 0.5
    G = 3.0

    solution, results, converged = solve(tau_w, tau_z, G)

    print("Solution status:", results["sol"].status)
    print("Solution message:", results["sol"].message)
    print("Convergence:", converged)
    print("Solution vector [T_C, T_D, Z_C, Z_D, w, p_D, L]:")
    print(solution)

    print("\nProduction Summary:")
    print(f"Sector C: T_prod = {results['T_C']:.4f}, Z_C = {results['Z_C']:.4f}")
    print(f"Sector D: T_prod = {results['T_D']:.4f}, Z_D = {results['Z_D']:.4f}")
    print(f"Common wage, w = {results['w']:.4f}, p_D = {results['p_D']:.4f}")
    print(f"Sector C output, F_C = {results['F_C']:.4f}")
    print(f"Sector D output, F_D = {results['F_D']:.4f}")

    print("\nHousehold Demands and Leisure:")
    for i in range(n):
        print(f"Household {i+1}: C = {results['C_agents'][i]:.4f}, D = {results['D_agents'][i]:.4f}, l = {results['l_agents'][i]:.4f}")

    print("\nAggregated Quantities:")
    print(f"Aggregate C = {results['agg_C']:.4f}")
    print(f"Aggregate D = {results['agg_D']:.4f}")
    print(f"Labor supply = {results['agg_labor']:.4f}")

    print("\nLump sum:")
    print(f"L = {results['L']:.4f}")

    print("\nFirm Profits:")
    print(f"Profit Sector C: {results['profit_C']:.4f}")
    print(f"Profit Sector D: {results['profit_D']:.4f}")

    print("\nHousehold Budget Constraints:")
    for i in range(n):
        print(f"Household {i+1}: Error = {results['budget_errors'][i]:.10f}")

    print(f"\nGood C Market Clearing Residual: {(results['agg_C'] + 0.5*G) - results['F_C']:.10f}")
    
    print("\nHousehold Utilities:")
    for i in range(n):
        print(f"Household {i+1}: Utility = {results['utilities'][i]:.4f}")
