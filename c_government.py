from types import SimpleNamespace
from scipy.optimize import minimize
import numpy as np
from a_hh import workerProblem

class govSWF():
    def __init__(self):
        """setup"""
        # 1. Define the parameter vector.
        parGov = self.parGov = SimpleNamespace()
        
        # 2. Set exogenous parameters.
        parGov.omega = 0.3   # inequality aversion parameter
        parGov.g = 1         # government spending requirement
        parGov.time = 1      # total time endowment
        
        # 3. Define solution set.
        solGov = self.solGov = SimpleNamespace()
        
        # 4. Initialize solution parameters (to be determined by optimization).
        self.solGov.tau_w = None  # optimal labor income tax
        self.solGov.tau_z = None  # optimal pollution tax
        self.solGov.l = None      # lump-sum transfer

    def swf(self, c, b, ell):
        """Social welfare function that calls the household's utility function."""
        # 1. Retrieve exogenous parameters.
        parGov = self.parGov
        
        # 2. Create a worker instance and compute utility.
        worker = workerProblem()
        
        # 3. Return the social welfare (weighted utility).
        return (1 / (1 - parGov.omega)) * worker.utility(c, b, ell)**(1 - parGov.omega)
    
    def government(self, w, phi, z):
        """Maximize welfare by choosing optimal taxes and transfers."""
        # 1. Retrieve solution set and parameter vector.
        solGov = self.solGov
        parGov = self.parGov
        
        # 2. Add endogenous parameters.
        parGov.w = w  # equilibrium wage
        parGov.z = z  # optimal pollution
        
        # 3. Add variable parameters.
        parGov.phi = phi  # share of polluting input in production
        
        # 4. Define the objective function (negative SWF).
        def obj(x):
            tau_w, tau_z, l = x  # decision variables
            worker = workerProblem()  # call the worker problem
            sol = worker.worker(phi=phi, tau=tau_w, w=w, pb=1, pc=1, l=l)  # optimal worker solution
            return -self.swf(sol.c, sol.b, sol.ell)
        
        # 5. Define bounds: tau_w and tau_z between 0 and 1; l nonnegative.
        bounds = [(0, 1), (0, 1), (0, None)]
        
        # 6. Additional equality constraint: government budget must balance.
        cons = [
            {'type': 'eq', 
             'fun': lambda x: x[0] * phi * w * (1 - workerProblem().worker(phi=phi, tau=x[0], w=w, pb=1, pc=1, l=x[2]).ell) + x[1] * z + x[2] - parGov.g}
        ]
        
        # 7. Initial guess (tau_w, tau_z, l).
        x0 = [0.2, 0.1, 0.1]  # start within the feasible region
        
        # 8. Solve using a constrained optimizer.
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        # 9. Save the solution.
        solGov.tau_w = res.x[0]
        solGov.tau_z = res.x[1]
        solGov.l = res.x[2]
        
        # 10. Print the solution.
        print(f'solution: tau_w={solGov.tau_w:.2f}, tau_z={solGov.tau_z:.2f}, l={solGov.l:.2f}')
        
        # 11. Return the solution set.
        return solGov

# Test the model.
model = govSWF()
model.government(w=11, phi=0.5, z=0.1)