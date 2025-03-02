from types import SimpleNamespace
from scipy.optimize import minimize_scalar
from a_hh import workerProblem
import numpy as np
from alt_b_firm_old import firmProblem
from scipy import optimize

class equilibirium():
    
    def __init__(self):
        self.parEq = SimpleNamespace()
        self.solEq = SimpleNamespace()
    
    def evaluate_equilibrium(self, p, w):
        parEq = self.parEq
        
        # Define an objective function for z.
        # It returns the squared difference between the guessed z and the firm's implied z.
        def obj_z(z):
            try:
                # Solve the household problem using z (here, for example, letting labor decision depend on z)
                hh = workerProblem()
                hhSol = hh.worker(phi=1, tau=0, w=w, pb=p, pc=p, l=0.25 * z)
                c = hhSol.c
                b = hhSol.b
                
                # Solve the firm problem with target output y = c + b.
                firm = firmProblem()
                firmSol = firm.solve_firm(w=8, tau_z=0.25, y=c+b)
                if firmSol is None:
                    # Return a high penalty if firm did not converge.
                    return 1e6
                # We want the firm's computed z to match our guess.
                return (firmSol.z - z)**2
            except Exception as e:
                # In case of any error, return a high penalty.
                return 1e6
        
        # First, do a coarse grid search over z to get a robust initial guess.
        z_grid = np.linspace(1e-3, 100, 1000)
        grid_vals = np.array([obj_z(z) for z in z_grid])
        best_index = np.argmin(grid_vals)
        z_initial = z_grid[best_index]
        
        # Now refine with a bounded minimizer.
        res = optimize.minimize_scalar(obj_z, bounds=(max(1e-3, 0.5*z_initial), 1.5*z_initial),
                                       method='bounded', options={'xatol': 1e-12})
        if not res.success:
            print("z-optimization did not converge:", res.message)
            z_star = z_initial  # fallback to grid solution
        else:
            z_star = res.x
        
        # With the converged z, re-solve the household and firm problems.
        hh = workerProblem()
        hhSol = hh.worker(phi=1, tau=0, w=w, pb=p, pc=p, l=5 * z_star)
        c = hhSol.c
        b = hhSol.b
        
        firm = firmProblem()
        firmSol = firm.solve_firm(w=8, tau_z=5, y=c+b)
        
        # Compute excess demands for diagnostics.
        g_m = (c + b) - firmSol.y       # goods market excess demand
        l_m = firmSol.t - (1 - hhSol.ell) # labor market excess demand
        parEq.g_m = g_m
        parEq.l_m = l_m
        
        print(f'excess goods market demand: {g_m:.2f}, excess labor market demand: {l_m:.2f}')
        return parEq
    
    def find_equilibrium(self):
        # Fix p as numeraire.
        p = 1.0
        
        # Define a univariate objective function on wage.
        def objective(w):
            par = self.evaluate_equilibrium(p, w)
            return par.g_m**2 + par.l_m**2
        
        # Choose wage search bounds.
        wage_lower = 0.1
        wage_upper = 50.0
        
        res = optimize.minimize_scalar(objective, bounds=(wage_lower, wage_upper),
                                       method='bounded', options={'xatol': 1e-12})
        self.solEq.w = res.x
        self.solEq.p = p
        print(f'equilibrium found: p = {self.solEq.p:.4f}, w = {self.solEq.w:.4f}')
        return self.solEq

# Example usage:
eq = equilibirium()
eq.find_equilibrium()

# Check hh problem separately.
model = workerProblem()
model.worker(phi=1, tau=0, w=16.2562, pb=1, pc=1, l=0)