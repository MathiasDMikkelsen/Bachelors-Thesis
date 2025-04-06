import numpy as np
import numba as nb
from scipy.optimize import differential_evolution
np.set_printoptions(formatter={'float_kind': lambda x: format(x, '.8f')})   # Adjusting output format to show 8 decimal places.

##############################################################################################################################################################
# 1) DEFINE THE MODEL FUNCTIONS (Household FOCs, Firm Production and FOCs, Market Clearing of Goods and Labor)
##############################################################################################################################################################

@nb.njit
def hh_focs(c, d, ell, p, w, alpha, beta, gamma, d0, mult):
    """
    This function computes the first-order conditions (FOCs) of the households and returns an array of residuals.
    Households have the utility function U = c^α + (d-d0)^β + ell^γ.
    Hence, the FOCs are:
      α·c^(α-1) = mult · p,
      β·(d-d0)^(β-1) = mult · p,
      γ·ell^(γ-1) = mult · w.
    Feasibility: c>0, d>d0, 0 < ell < 1.
    If any feasibility constraints are violated, the function returns an array with large residuals to penalize this in the optimization algorithm.
    """
    if c <= 0.0 or d <= d0 or ell <= 0.0 or ell >= 1.0:     # Check if constraints are violated.
        return np.array([1e6, 1e6, 1e6])                    # Return an array with large residuals if constraints are violated.
    foc_c = alpha * (c**(alpha-1)) - mult * p               # FOC residual for c.
    foc_d = beta * ((d-d0)**(beta-1)) - mult * p            # FOC residual for d.
    foc_ell = gamma * (ell**(gamma-1)) - mult * w           # FOC residual for ell.
    return np.array([foc_c, foc_d, foc_ell])                # Return array with FOC residuals if feasibility constraints are satisfied.

@nb.njit
def firm_production(t, z, epsilon, r):
    """
    This function computes and returns the output of the firm given inputs t and z.
    The firm has a CES production function y = [ε·t^r + (1-ε)·z^r]^(1/r) with constant returns to scale.
    Feasibility: inside>0.
    If the inside of the root is nonpositive, the function returns 0.
    """
    inside = epsilon*(t**r) + (1.0 - epsilon)*(z**r)        # Calculate the inside of the root in the production function.
    if inside <= 0.0:                                       # Check if the inside of the root is less than or equal to 0.
        return 0.0                                          # Return 0 if the constraint is violated.
    return inside**(1.0/r)                                  # Return the output of the firm if the constraint is satisfied.

@nb.njit
def firm_focs(t, z, p, w, tau_z, epsilon, r):
    """
    This function computes the first-order conditions (FOCs) of the firm using the firm_production function and returns an array of residuals.
    The firm's FOCs are:
      p * (dy/dt) - w = 0,
      p * (dy/dz) - τ_z = 0.
    Feasibility: y>0.
    If the output of the firm is nonpositive, the function returns an array with large residuals to penalize this in the optimization algorithm.
    """
    y = firm_production(t, z, epsilon, r)                   # Calculate the output of the firm using the firm_production function.
    if y <= 0.0:                                            # Check if the output of the firm is nonpositive.
        return np.array([1e6, 1e6])                         # Return an array large residuals if the constraint is violated.
    dy_dt = epsilon*(t**(r-1))*(y**(1.0-r))                 # The derivative of y with respect to t.
    dy_dz = (1.0 - epsilon)*(z**(r-1))*(y**(1.0-r))         # The derivative of y with respect to z.
    foc_t = p * dy_dt - w                                   # FOC residual for t.
    foc_z = p * dy_dz - p*tau_z                             # FOC residual for z. Multiplying by p to maintain p as numeraire.
    return np.array([foc_t, foc_z])                         # Return array with FOC residuals if the feasibility constraint is satisfied.

@nb.njit
def market_clearing(c, d, ell, t, z, p, w, epsilon, r):
    """
    This function calculates the excess demand in the goods and labor markets and returns this in an array.
    The market clearing conditions are:
      Goods: (c + d) - y(t,z) = 0,
      Labor: (1 - ell) - t = 0  (i.e., t = 1 - ell).
    """
    y = firm_production(t, z, epsilon, r)                  # Calculate the output of the firm using the firm_production function.
    f_goods = (c + d) - y                                  # Excess demand in the goods market.
    f_labor = (1.0 - ell) - t                              # Excess demand in the labor market.
    return np.array([f_goods, f_labor])                    # Return array with excess demand in the goods and labor markets.

##############################################################################################################################################################
# 2) CREATE THE SYSTEM OF EQUATIONS
##############################################################################################################################################################
def full_system(x, params):
    """
    This function combines the residual arrays into one 7-dimensional residual vector so it can be minimized by the differential evolution algorithm.
    The vector consists of residuals from:
      3 household FOCs,
      2 firm FOCs,
      2 market clearing equations.
    We have chosen p as numeraire and is thus not included in the system and instead included as a parameter. As a result, the budget constraint can be excluded from the system.
    """
    c, d, ell, t, z, w, mult = x                           # Creates an input vector with the endogenous variables that are to be solved for.
    p = params['p']                                        # Sets p as the numeraire value.

    alpha   = params['alpha']                              # Defines the parameter alpha.
    beta    = params['beta']                               # Defines the parameter beta.
    gamma   = params['gamma']                              # Defines the parameter gamma.
    d0      = params['d0']                                 # Defines the parameter d0.
    tau_z   = params['tau_z']                              # Defines the parameter tau_z.
    epsilon = params['epsilon']                            # Defines the parameter epsilon.
    r       = params['r']                                  # Defines the parameter r.

    hh_res   = hh_focs(c, d, ell, p, w, alpha, beta, gamma, d0, mult)   # Defines hh_res as an array with the three household FOC residuals.
    firm_res = firm_focs(t, z, p, w, tau_z, epsilon, r)                 # Defines firm_res as an array with the two firm FOC residuals.
    mkt_res  = market_clearing(c, d, ell, t, z, p, w, epsilon, r)       # Defines mkt_res as an array with the two market clearing residuals.

    return np.concatenate((hh_res, firm_res, mkt_res))     # Returns a concatenated array with all residuals.

##############################################################################################################################################################
# 3) DEFINE OBJECTIVE FUNCTION (Sum of Squared Residuals)
##############################################################################################################################################################
def objective(x, params):
    """
    This function defines the objective function that is to be minimized by the differential evolution algorithm.
    The objective function returns the sum of squared residuals from the full system of equations.
    """
    res = full_system(x, params)                          # Calculates the residual vector from the full system of equations and saves it as res.
    return np.sum(res**2)                                 # Returns the sum of squared residuals.

##############################################################################################################################################################
# 4) DEFINE MAIN FUNCTION THAT FINDS THE EQUILIBRIUM USING DIFFERENTIAL EVOLUTION
##############################################################################################################################################################
def main():
    """
    This function defines the main function that finds the equilibrium of the model using the differential evolution algorithm.
    The function sets the numeraire value, defines the parameters as a dictionary, sets the bounds for the endogenous variables, defines the objective function for differential evolution, and runs the global optimizer.
    It then returns the results of the optimization and computes the equilibrium output and profit from the firm.
    """
    numeriare_value = 2                                   # Sets the numeraire value.

    params = {                                            # Defines the parameters as a dictionary.
        'alpha':   0.7,
        'beta':    0.2,
        'gamma':   0.2,
        'd0':      0.5,
        'tau_z':   0.2,
        'epsilon': 0.6,
        'r':       0.5,
        'p':       numeriare_value
    }

    bounds = [                                            # Set bounds that are generous but keep variables in reasonable ranges.
        (1e-6, 100.0),                                    # Bounds for c.
        (params['d0'], 100.0),                            # Bounds for d.
        (1e-6, 0.9999),                                   # Bounds for ell.
        (1e-6, 1.0),                                      # Bounds for t.
        (1e-6, 100.0),                                    # Bounds for z.
        (1e-6, 100.0),                                    # Bounds for w.
        (1e-6, 20.0)                                      # Bounds for the Lagrange multiplier on the household budget constraint.
    ]

    def f_obj(x):                                         # Define the objective function for differential evolution that only takes the endogenous variables as input.
        return objective(x, params)                       # Returns the sum of squared residuals from the full system of equations.

    result = differential_evolution(                      # Runs the differential evolution algorithm to minimize the objective function and find the equilibrium.
        f_obj,                                            # Calls the objective function as the function we are minimizing.
        bounds,                                           # Imposes the bounds on the endogenous variables.
        strategy='best1bin',                              # Chooses the best1bin strategy (very common strategy) for the differential evolution algorithm.
        maxiter=2000,                                     # Sets the maximum number of iterations to 2000. If a solution is not found within this number of iterations, the algorithm stops and returns the best solution found so far.
        popsize=30,                                       # Sets the population size to 30. The population size is the number of candidate solutions that are evolved in each iteration of the algorithm.
        tol=1e-7,                                         # Sets the tolerance for convergence. The algorithm stops if the standard deviation of the objective function values in the population falls below the tolerance level.
        mutation=(0.5, 1.0),                              # Sets the mutation factor to be drawn randomly over a range for each generation. The mutation factor controls the amplification of the differential variation between two randomly selected members that are added to the best current member to create the mutant member.
        recombination=0.7,                                # Sets the recombination constant to 0.7. The recombination constant controls the probability that each element in the trial vector is taken from the mutant member rather than the current best member. At least one parameter is taken from the mutant member.
        disp=True                                         # Prints a message each time the solution improves during the optimization process.
    )
    print("\n=== Differential Evolution Result ===")
    print("Converged:", result.success)
    print("Message: ", result.message)
    print("Minimum sum-of-squared errors:", result.fun)

    x_sol = result.x
    c, d, ell, t, z, w, mult = x_sol

    print("\n--- Endogenous Variables ---")
    print(f"c (Clean Goods Consumption): {c:.8f}")
    print(f"d (Dirty Goods Consumption): {d:.8f}")
    print(f"ell (Leisure): {ell:.8f}")
    print(f"t (Labor Input): {t:.8f}")
    print(f"z (Pollution Input): {z:.8f}")
    print(f"w (Wage Rate): {w:.8f}")
    print(f"mult (Lagrange Multiplier): {mult:.8f}")

    p = params['p']
    epsilon = params['epsilon']
    r = params['r']
    tau_z = params['tau_z']

    y = firm_production(t, z, epsilon, r)
    # Equilibrium profit: Revenue minus cost of labor and z.
    profit = p * y - w * t - p*tau_z * z

    print("\n--- Firm ---")
    print(f"y (Output): {y:.8f}")
    print(f"Profit: {profit:.8f}")

    hh_res   = hh_focs(c, d, ell, p, w, params['alpha'], params['beta'], params['gamma'], params['d0'], mult)
    firm_res = firm_focs(t, z, p, w, tau_z, epsilon, r)
    mkt_res  = market_clearing(c, d, ell, t, z, p, w, epsilon, r)

    # Calculate budget constraint residual
    budget_res = p*c + p*d - w * (1 - ell) - tau_z*z*p

    print("\n--- Residuals ---")
    print(f"Household FOC c: {hh_res[0]:.3e}")
    print(f"Household FOC d: {hh_res[1]:.3e}")
    print(f"Household FOC ell: {hh_res[2]:.3e}")
    print(f"Firm FOC t: {firm_res[0]:.3e}")
    print(f"Firm FOC z: {firm_res[1]:.3e}")
    print(f"Market Clearing Goods: {mkt_res[0]:.3e}")
    print(f"Market Clearing Labor: {mkt_res[1]:.3e}")
    print(f"Budget Constraint: {budget_res:.3e}")

if __name__=="__main__":
    main()