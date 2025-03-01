import numpy as np
from types import SimpleNamespace
from scipy import optimize

class FirmProblem:
    def __init__(self):
        """Sets up default parameters and solution storage."""
        self.par = SimpleNamespace()

        # Example calibration:
        self.par.r = 0.5        # elasticity exponent (try 0.5 before 0.05)
        self.par.epsilon = 0.5   # CES weight on t
        self.par.w = 6         # cost of t
        self.par.tau_z = 12     # cost of z
        self.par.p = 6.0        # price of output

        # Where we'll store results
        self.sol = SimpleNamespace()


    def production(self, t, z):
        p = self.par
        inside = p.epsilon*(t**p.r) + (1 - p.epsilon)*(z**p.r)
        return inside if inside>0 else 0.0

    def solve_firm(self):
        p = self.par
        # solve optimization problem
        obj = lambda x: -(p.p * self.production(x[0], x[1]) - p.w * x[0] - p.tau_z * x[1])
        x0 = [99, 99]
        res = optimize.minimize(obj, x0, method='L-BFGS-B')
        t_opt, z_opt = res.x

        # Store and print
        self.sol.t = t_opt
        self.sol.z = z_opt

        # result
        print(f"t = {t_opt:.4f}, z = {z_opt:.4f}")

        return self.sol

# ========== TEST THE CODE ==========

model = FirmProblem()

# Optional: override very small 'r' if you want to see the solver get stuck
# model.par.r = 0.05

model.solve_firm()