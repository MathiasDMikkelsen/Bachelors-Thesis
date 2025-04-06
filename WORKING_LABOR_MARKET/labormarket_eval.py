import numpy as np
from WORKING_LABOR_MARKET.inner_labormarkets import solve  # Make sure inner_solver.py contains the updated solve function

# a. Set fiscal parameters
tau_w = np.array([0.1, 0.12, 0.15, 0.18, 0.2])
g = 5.0

# b. Create a range of tau_z values
tau_z_values = np.linspace(0.1, 3.0, 100)

# c. Track failures
failed_tau_z = []

# d. Loop over tau_z values
for tau_z in tau_z_values:
    _, _, converged = solve(tau_w, tau_z, g)
    if not converged:
        failed_tau_z.append(tau_z)
        print(f"❌ Solver failed for tau_z = {tau_z:.3f}")
    else:
        print(f"✅ Solver succeeded for tau_z = {tau_z:.3f}")

# e. Summary
print("\n========== Summary ==========")
if failed_tau_z:
    print("Solver did not converge for the following tau_z values:")
    for tz in failed_tau_z:
        print(f"  tau_z = {tz:.3f}")
else:
    print("Solver converged for all tau_z values in the range.")
