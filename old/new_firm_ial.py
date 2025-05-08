import numpy as np
from scipy.optimize import fsolve

def system_equations(x, r, epsilon, w, tau_z, y):
    """
    x is a vector [t, z, lambda], where:
      - t and z are input quantities,
      - lambda is the Lagrange multiplier from the production constraint.
      
    The system is:
      (1) w - lambda * epsilon * r * t^(r-1) = 0,
      (2) tau_z - lambda * (1-epsilon) * r * z^(r-1) = 0,
      (3) epsilon * t^r + (1-epsilon) * z^r - y^r = 0 (with y normalized to 1).
    """
    t, z, lam = x
    eq1 = w - lam * epsilon * r * t**(r-1)      
    eq2 = tau_z - lam * (1 - epsilon) * r * z**(r-1)
    eq3 = epsilon * t**r + (1 - epsilon) * z**r - y**r
    return [eq1, eq2, eq3]

# Set parameter values
r = 0.5
epsilon = 0.5
w = 2.0       # cost of t
tau_z = 1.0   # cost of z
y = 1.0       # normalized output

# For r = 0.5 and epsilon = 0.5, the ratio condition implied by the FOCs is:
#   (t/z)^(1-r) = (w/tau_z)*((1-epsilon)/epsilon).
# With r=0.5, 1-r=0.5 so that: (t/z)^0.5 = w/tau_z.
# Equivalently, z/t = (tau_z/w)^2.
# For w=2 and tau_z=1, we expect z/t = (1/2)^2 = 0.25.
# A reasonable guess is then, say, t ≈ 1.8 and z ≈ 0.45, with lambda around 10.
initial_guess = [1.8, 0.45, 10.0]

# Solve the system using fsolve.
solution, infodict, ier, mesg = fsolve(system_equations, initial_guess, args=(r, epsilon, w, tau_z, y), full_output=True)

if ier == 1:
    t_opt, z_opt, lam_opt = solution
    print("Converged solution:")
    print(f"  t = {t_opt:.6f}")
    print(f"  z = {z_opt:.6f}")
    print(f"  lambda = {lam_opt:.6f}")
    # Compute and print residual errors of the system:
    residuals = system_equations(solution, r, epsilon, w, tau_z, y)
    print("Residual errors:", np.array(residuals))
else:
    print("Solution did not converge:", mesg)