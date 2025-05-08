import numpy as np
from scipy.optimize import minimize, Bounds, NonlinearConstraint
import functools

# Assumes you have 'solver_constlump.py' with a function:
#   solve(tau_w, tau_z, G, n=5)
# returning: (utilities, agg_polluting, converged, c, d, ell, w, p_d, l_val, tax_rev)
import solver_constlump as solver

np.set_printoptions(suppress=True, precision=8)

######################################################
# 1) Parameters (unchanged)
######################################################
xi = 2.0
theta = 1.0
n = 5
G = 2.0
phi = np.array([0.03, 0.0825, 0.141, 0.229, 0.5175]) 
# ~ [0.15, 0.4125, 0.705, 1.145, 2.5875]
l_fixed = np.array([0.2]*n)  # lumpsum = 0.2 each => total lumpsum = 1.0

# We'll define x = [tau_w_1, ..., tau_w_5, tau_z].
dim = n + 1

######################################################
# 2) Cache calls to solver.solve(...) to avoid repeats
######################################################
@functools.lru_cache(None)
def cached_solve(tau_w_tuple, tau_z, G_val, n_val):
    """
    Caches results keyed by (tau_w_tuple, tau_z, G_val, n_val).
    Because lists aren't hashable, we convert tau_w to a tuple.
    """
    tau_w = np.array(tau_w_tuple)
    return solver.solve(tau_w, tau_z, G_val, n=n_val)

def call_solver(x):
    tau_w = x[:n]
    tau_z = x[n]
    return cached_solve(tuple(tau_w), tau_z, G, n)

######################################################
# 3) Objective Function (swf_obj) + Domain Checks
######################################################
def objective_fun(x):
    """
    Returns -welfare for minimization => max welfare.
    If solver fails, returns big penalty (1e6).
    """
    utilities, agg_polluting, converged, c, d, ell, w, p_d, l_val, tax_rev = call_solver(x)
    
    # If model not converged, penalize
    if not converged:
        return 1e6
    
    # If any utility is NaN/Inf => penalty
    if np.isnan(utilities).any() or np.isinf(utilities).any():
        return 1e6

    # Then compute welfare
    welfare = np.sum(utilities) - xi*(agg_polluting)
    if not np.isfinite(welfare):
        return 1e6

    return -welfare


def objective_grad(x, eps=1e-4):
    """
    Forward-difference approximation to the gradient (Jacobian of objective).
    We'll do a safe call to objective_fun for each dimension + small eps.
    """
    f0 = objective_fun(x)
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_pert = np.copy(x)
        x_pert[i] += eps
        f_pert = objective_fun(x_pert)
        if np.isnan(f_pert) or np.isinf(f_pert):
            grad[i] = 0.0
        else:
            grad[i] = (f_pert - f0)/eps
    return grad

######################################################
# 4) IC Constraints (U_i - U_i_j >= 0) + Domain Checks
######################################################
def ic_constraints_fun(x):
    """
    Returns array of length n*(n-1). Must be >= 0 => no incentive to masquerade.
    We'll do log domain checks and never return NaN or inf.
    """
    utilities, agg_polluting, converged, c, d, ell, w, p_d, l_val, tax_rev = call_solver(x)
    
    # If no converge => all constraints fail
    if not converged:
        return -np.ones(n*(n-1)) * 1e3
    
    # If any of the "true" utilities are NaN/inf => penalize
    if np.isnan(utilities).any() or np.isinf(utilities).any():
        return -np.ones(n*(n-1)) * 1e3

    alpha, beta, gamma = 0.7, 0.2, 0.2
    d0 = 0.02
    T = 1.0
    g_list = []

    # Income for type j
    # x[j] = tau_w[j], so (1 - x[j]) is (1 - tau_w[j])
    I = np.zeros(n)
    for j in range(n):
        I[j] = (T - ell[j])*(1.0 - x[j])*phi[j]*w 

    for i in range(n):
        U_i = utilities[i]  # "truthful" utility
        # If that is invalid => penalize
        if not np.isfinite(U_i):
            # produce n-1 negative constraints
            g_list.extend([-1e3]*(n-1))
            continue

        for j in range(n):
            if i == j: 
                continue
            
            c_j = c[j]
            d_j = d[j]
            denom = (1.0 - x[j])*phi[i]*w
            if denom <= 1e-12:
                # invalid => big negative
                g_list.append(-1e3)
                continue

            ell_i_j = T - I[j]/denom
            # domain check
            if ell_i_j <= 1e-12 or c_j <= 1e-12 or d_j <= d0+1e-12:
                # means log(...) would fail => penalty
                U_i_j = -1e3
            else:
                # safe logs
                val_c = alpha*np.log(c_j)
                val_d = beta*np.log(d_j - d0)
                val_ell = gamma*np.log(ell_i_j)
                U_i_j = val_c + val_d + val_ell

            # constraint => U_i >= U_i_j => (U_i - U_i_j) >= 0
            g_list.append(U_i - U_i_j)

    return np.array(g_list)

def ic_constraints_jac(x, eps=1e-4):
    """
    Forward-diff approximation for the constraints' Jacobian.
    We'll be sure to handle domain errors carefully as well.
    Return shape = (m, dim), where m=n*(n-1), dim=n+1.
    """
    g0 = ic_constraints_fun(x)
    m = len(g0)
    jac = np.zeros((m, dim))
    for j in range(dim):
        x_pert = np.copy(x)
        x_pert[j] += eps
        g_pert = ic_constraints_fun(x_pert)
        dg = (g_pert - g0)/eps
        # If we see NaN/Inf, keep it finite
        mask = np.isnan(dg) | np.isinf(dg)
        dg[mask] = 0.0
        jac[:, j] = dg
    return jac

######################################################
# 5) Build NonlinearConstraint for the IC constraints
######################################################
ic_constraint = NonlinearConstraint(
    fun=ic_constraints_fun,
    lb=0.0,
    ub=np.inf,
    jac=ic_constraints_jac
    # hess=None  # omitted for now
)

######################################################
# 6) Bounds and Minimization with trust-constr
######################################################
if __name__ == "__main__":
    # initial guess
    x0 = np.array([0.05]*n + [0.5])

    lb = [-5.0]*n + [0.01]
    ub = [5.0]*n + [100.0]
    bds = Bounds(lb, ub)

    res = minimize(
        fun=objective_fun,
        x0=x0,
        method='trust-constr',
        jac=objective_grad,
        bounds=bds,
        constraints=[ic_constraint],
        options={
            'verbose': 3,
            'maxiter': 300,
            'gtol': 1e-6,
            'xtol': 1e-6,
            'barrier_tol': 1e-6,
        }
    )

    print("\nRESULT:")
    print("Success:", res.success)
    print("Status:", res.status)
    print("Message:", res.message)
    print("Final x:", res.x)
    print("Objective:", res.fun)
    # Check final constraints
    g_final = ic_constraints_fun(res.x)
    print("IC constraints at final x:\n", g_final)
    print("Min of constraints (should be >= 0 if feasible):", g_final.min())