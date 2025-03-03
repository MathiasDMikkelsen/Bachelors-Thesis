import numpy as np
from types import SimpleNamespace
from a_hh import workerProblem
from alt_b_firm_old import firmProblem

class EquilibriumGridSearch:
    def __init__(self):
        self.sol = SimpleNamespace()
    
    def equilibrium_residuals(self, w, z):
        """
        For a given wage w and linking variable z:

        1) Solve the household problem with labor = 0.25*z.
           - Let c = consumption, b = saving, ell = leisure.
        2) Household's 'target output' is y_target = c + b.
        3) Solve the firm problem with target y = y_target (and the same wage w!).
           - Let firmSol.t = labor demand, firmSol.z = linking var, firmSol.y = production.

        We define three equilibrium residuals:
          F1 = firmSol.z - z                         (linking consistency)
          F2 = (c + b) - firmSol.y                   (goods market clearing)
          F3 = firmSol.t - (1 - hhSol.ell)           (labor market clearing)

        Returns (F1, F2, F3).
        """
        # Solve the household problem.
        hh = workerProblem()
        hhSol = hh.worker(phi=1, tau=0, w=w, pb=1.0, pc=1.0, l=0.25 * z)
        
        c = hhSol.c
        b = hhSol.b
        ell = hhSol.ell
        
        y_target = c + b  # household's total usage (consumption + saving)
        
        # Solve the firm problem with the same wage w (NOT a fixed w=8).
        firm = firmProblem()
        firmSol = firm.solve_firm(w=w, tau_z=0.25, y=y_target)
        
        # If firmSol fails or is None, return large residuals so it's penalized.
        if firmSol is None or not hasattr(firmSol, 'z') or not hasattr(firmSol, 'y') or not hasattr(firmSol, 't'):
            return 1e6, 1e6, 1e6
        
        # Define the three residuals:
        F1 = firmSol.z - z                 # linking consistency
        F2 = (c + b) - firmSol.y           # goods market clearing
        F3 = firmSol.t - (1.0 - ell)       # labor market clearing

        return F1, F2, F3
    
    def find_equilibrium(self):
        """
        Brute-force 2D grid search:
          - We iterate over wage (w) in w_grid, linking variable (z) in z_grid.
          - At each grid point, compute (F1, F2, F3) from equilibrium_residuals(...).
          - Objective = F1^2 + F2^2 + F3^2.
          - Choose the (w,z) that minimize this objective.
        
        Because we have 3 conditions but only 2 unknowns, we typically can't get all
        residuals to zero. We'll pick the best approximation on the grid.
        """
        # Define grids (adjust ranges/resolutions as needed).
        w_grid = np.linspace(0.5, 20.0, 25)    # 25 points for wage
        z_grid = np.linspace(1e-3, 50.0, 25)   # 25 points for linking var
        
        best_val = np.inf
        best_w = None
        best_z = None
        best_F = (None, None, None)
        
        # Loop over the (w, z) grid.
        for w in w_grid:
            for z in z_grid:
                F1, F2, F3 = self.equilibrium_residuals(w, z)
                val = F1**2 + F2**2 + F3**2
                if val < best_val:
                    best_val = val
                    best_w = w
                    best_z = z
                    best_F = (F1, F2, F3)
        
        self.sol.w = best_w
        self.sol.z = best_z
        self.sol.obj_value = best_val
        self.sol.F1, self.sol.F2, self.sol.F3 = best_F
        
        print("===== 2D Grid Search Equilibrium =====")
        print(f"Best wage (w): {best_w:.4f}")
        print(f"Best linking variable (z): {best_z:.4f}")
        print(f"Residuals: F1 = {best_F[0]:.4e}, F2 = {best_F[1]:.4e}, F3 = {best_F[2]:.4e}")
        print(f"Objective (F1^2 + F2^2 + F3^2) = {best_val:.4e}")
        
        return self.sol

if __name__ == "__main__":
    eq_solver = EquilibriumGridSearch()
    solution = eq_solver.find_equilibrium()