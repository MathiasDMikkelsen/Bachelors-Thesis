from types import SimpleNamespace
import numpy as np
from scipy import optimize

class firmProblem:
    
    def __init__(self):
        # define parameter vector
        self.parFirm = SimpleNamespace()
        # define exogenous parameters
        self.parFirm.r = 0.5         # elasticity exponent
        self.parFirm.epsilon = 0.5   # CES weight on t
        # define solution vector
        self.sol = SimpleNamespace()
        
    def inside(self, t, z):
        parFirm = self.parFirm
        inside_val = parFirm.epsilon * (t ** parFirm.r) + (1 - parFirm.epsilon) * (z ** parFirm.r)
        return inside_val if inside_val > 0 else 0.0

    def foc_errors(self, t, z):
        parFirm = self.parFirm
        # Compute production output using the CES function
        f = self.inside(t, z)**(1/parFirm.r)
        
        # avoid division by zero or negative production
        if self.inside(t, z) <= 0:
            return np.array([np.nan, np.nan])
        
        # derivatives w.r.t. t and z
        df_dt = parFirm.epsilon * (t ** (parFirm.r - 1)) * (f ** (1 - parFirm.r))
        df_dz = (1 - parFirm.epsilon) * (z ** (parFirm.r - 1)) * (f ** (1 - parFirm.r))
        
        # first-order condition errors
        err_t = parFirm.p * df_dt - parFirm.w
        err_z = parFirm.p * df_dz - parFirm.tau_z
        
        return np.array([err_t, err_z])
    
    def solve_firm_bisection_t(self, w, tau_z, p):
        # set variable parameters
        self.parFirm.w = w             # cost of t
        self.parFirm.tau_z = tau_z     # cost of z
        self.parFirm.p = p             # price of output

        # For each t, solve for z (minimize z's FOC error) then return t's error (df/dt error)
        def t_error(t):
            # inner minimization: choose z to minimize squared error on z's first order condition
            def obj_z(z):
                foc = self.foc_errors(t, z[0])
                return foc[1]**2
            z0 = [0.5]
            res_z = optimize.minimize(obj_z, z0, bounds=[(1e-6, None)], method='L-BFGS-B',
                                        options={'ftol':1e-12, 'gtol':1e-12})
            if not res_z.success:
                # if z cannot be solved, return a large error
                return np.inf
            z_val = res_z.x[0]
            foc = self.foc_errors(t, z_val)
            # We now return the t FOC error
            return foc[0]
        
        # Find a bracket region for t where t_error changes sign. Adjust these limits as needed.
        t_min = 1e-6
        t_max = 1.0
        # Scan a coarse grid to find a sign change.
        t_grid = np.linspace(t_min, t_max, 1000)
        bracket_found = False
        for i in range(len(t_grid)-1):
            f1 = t_error(t_grid[i])
            f2 = t_error(t_grid[i+1])
            if f1 * f2 < 0:
                t_lower = t_grid[i]
                t_upper = t_grid[i+1]
                bracket_found = True
                break
        if not bracket_found:
            raise RuntimeError("No sign change found in t_error. Try redefining t_grid limits.")
        
        sol_t = optimize.root_scalar(t_error, bracket=[t_lower, t_upper], method='bisect', xtol=1e-12)
        t_sol = sol_t.root

        # Now, for the found t_sol, obtain the corresponding z:
        def obj_z(z):
            foc = self.foc_errors(t_sol, z[0])
            return foc[1]**2
        z0 = [0.5]
        res_z = optimize.minimize(obj_z, z0, bounds=[(1e-6, None)], method="L-BFGS-B",
                                  options={'ftol':1e-12, 'gtol':1e-12})
        if not res_z.success:
            raise RuntimeError("Failed to minimize z's FOC for t = %.8f" % t_sol)
        z_sol = res_z.x[0]
        
        # Save the solution
        self.sol.t = t_sol
        self.sol.z = z_sol
        self.sol.y = self.inside(t_sol, z_sol)**(1/self.parFirm.r)
        
        optimal_profit = self.parFirm.p * self.sol.y - self.parFirm.w * self.sol.t - self.parFirm.tau_z * self.sol.z
        print(f"Solution: t = {t_sol:.8f}, z = {z_sol:.8f}")
        print(f"Optimal profit: {optimal_profit:.8e}")
        return self.sol

# Test the bisection-based approach for t
model = firmProblem()
model.solve_firm_bisection_t(w=5, tau_z=2, p=10)
errors = model.foc_errors(model.sol.t, model.sol.z)
print(f"FOC errors: df/dt error = {errors[0]:.8e}, df/dz error = {errors[1]:.8e}")