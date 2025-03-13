import numpy as np   
import importlib                                                   
from scipy.optimize import minimize_scalar
import new2_ial 
importlib.reload(new2_ial)

# Parameters for the SWF function
xi = 5.0
theta = 1.0

def swf_obj(tau_z):
    # Unpack three values now
    utilities, aggregate_polluting, first_converged = new2_ial.main_solve_for_tau(tau_z, n=5)
    # Print convergence status for this tau_z candidate
    print(f"\nCandidate tau_z = {tau_z:.4f} | First Optimizer Convergence: {first_converged}")
    if not first_converged:
        # Penalize non-convergence.
        return 1e6
    welfare = np.sum(utilities) - xi * aggregate_polluting**theta
    return -welfare  # Negative because we use a minimizer.

# Optimize over a reasonable range of tau_z values.
res = minimize_scalar(swf_obj, bounds=(0.1, 100.0), method='bounded')
opt_tau_z = res.x
max_welfare = -res.fun

print("\n=== Social Welfare Optimization ===")
print("Optimal tau_z:", opt_tau_z)
print("Maximized Social Welfare:", max_welfare)
print("Optimizer Convergence:", res.success)