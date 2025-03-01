import numpy as np
from types import SimpleNamespace
from scipy import optimize

class FirmProblem:
    def __init__(self):
        """Sets up default parameters and solution storage."""
        self.par = SimpleNamespace()

        # Example calibration:
        self.par.r = 0.25         # elasticity exponent (try 0.5 before 0.05)
        self.par.epsilon = 0.5   # CES weight on t
        self.par.w = 6       # cost of t
        self.par.tau_z = 12     # cost of z
        self.par.p = 12.0        # price of output

        # Where we'll store results
        self.sol = SimpleNamespace(t=np.nan, z=np.nan, y=np.nan,
                                   pi=np.nan, foc_error=np.nan)

    def production(self, t, z):
        """CES production function f(t,z) = [ eps*t^r + (1-eps)*z^r ]^(1/r)."""
        p = self.par
        # If t=0 or z=0, we allow it but must avoid negative powers for r<1
        # => We'll handle that carefully in partial derivatives below
        inside = p.epsilon*(t**p.r) + (1 - p.epsilon)*(z**p.r)
        return inside**(1.0/p.r) if inside>0 else 0.0

    def partial_t(self, t, z):
        """df/dt for the CES aggregator, assuming t>0, z>0, inside>0."""
        p = self.par
        fval = self.production(t,z)
        if fval<=0 or t<=0:
            return 0.0  # or define a large penalty if you prefer
        # Standard CES partial:
        return p.epsilon * (t**(p.r-1)) * (fval**(1-p.r))

    def partial_z(self, t, z):
        """df/dz for the CES aggregator."""
        p = self.par
        fval = self.production(t,z)
        if fval<=0 or z<=0:
            return 0.0
        return (1 - p.epsilon) * (z**(p.r-1)) * (fval**(1-p.r))

    def foc_error(self, t, z):
        """
        Sum of squared FOCs:
          FOC wrt t: p * df/dt - w = 0
          FOC wrt z: p * df/dz - tau_z = 0
        Returns (err1^2 + err2^2).
        """
        p = self.par
        # If negative inputs, huge penalty so solver stays away
        if t<0 or z<0:
            return 1e12

        mp_t = self.partial_t(t,z)
        mp_z = self.partial_z(t,z)

        err1 = p.p * mp_t - p.w
        err2 = p.p * mp_z - p.tau_z
        return err1**2 + err2**2

    def solve_firm(self):
        """
        Minimize sum of squared FOCs subject to t>=0, z>=0.
        We use SLSQP so we can specify constraints directly.
        """
        
        #self.par.w = w
        #self.par.p = p
        

        f = lambda t,z: self.foc_error(t, z) 
        obj = lambda x: f(x[0], x[1])
        x0 = [2, 3]
        bounds = ((0.1, None), (0.1, None))
        res = optimize.minimize(obj,x0,bounds=bounds,method='L-BFGS-B')
        
        t_opt, z_opt = res.x
        # Evaluate final FOC error
        final_err = self.foc_error(t_opt, z_opt)

        # Compute production and profit
        y_opt = self.production(t_opt, z_opt)
        pi_opt = self.par.p*y_opt - self.par.w*t_opt - self.par.tau_z*z_opt

        # Store and print
        self.sol.t = t_opt
        self.sol.z = z_opt
        self.sol.y = y_opt
        self.sol.pi = pi_opt
        self.sol.foc_error = final_err

        print("\n--- Results ---")
        print("Success =", res.success, "| Message =", res.message)
        print(f"t = {t_opt:.4f}, z = {z_opt:.4f}")
        print(f"Production f(t,z) = {y_opt:.4f}")
        print(f"Profit = {pi_opt:.4f}")
        print(f"Sum of squared FOC errors = {final_err:.6g}")

        return self.sol

# ========== TEST THE CODE ==========

model = FirmProblem()

# Optional: override very small 'r' if you want to see the solver get stuck
# model.par.r = 0.05

model.solve_firm()