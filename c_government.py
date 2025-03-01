from types import SimpleNamespace
from scipy.optimize import minimize
import numpy as np
from scipy import optimize
from a_hh import workerProblem
from b_firm import firmProblem

class govSWF():
    def __init__(self):
        """ setup"""
        
        # a. define vector of parameters
        parGov = self.parGov = SimpleNamespace()

        # b. exogenous parameters
        parGov.omega = 0.3   # inequality aversion parameter
        parGov.g = 1         # government spending requirement
        parGov.time = 1      # total time endowment
        
        # d. define solution set
        solGov = self.solGov = SimpleNamespace()
        
        # e. solution parameters (to be determined by optimization)
        self.solGov.tau_w = None     # optimal labor income tax
        self.solGov.tau_z = None     # optimal pollution tax
        self.solGov.l = None         # lump-sum transfer (isolated via the budget constraint) 

    def swf(self, c, b, ell):
        """Social welfare function that calls the household’s utility function"""

        # retrieve exogenous parameters
        parGov = self.parGov
        
        # creating a worker instance and computing utility
        worker = workerProblem()
        
        # returning the welfare amount
        return (1 / (1 - parGov.omega))*worker.utility(c, b, ell)**(1-parGov.omega)
    
    def government(self, w, phi, z):
        """ maximize welfare """
        
        # a. retrieve solution set and parameter vector
        solGov = self.solGov
        parGov = self.parGov
        
        # b. add endogenous parameters to parameter vector
        parGov.w = w  # equilibrium wage
        parGov.z = z # optimal pollution
        
        # c. add variable parameters to parameter vector
        parGov.phi = phi  # share of polluting input in production
        
        # d. define objective function as negative welfare with budget constraint inserted
        def obj(x):
            tau_w, tau_z, l = x # define variables
            worker = workerProblem()
            sol = worker.worker(phi=phi, tau=tau_w, w=w, pb=1, pc=1, l=l)
            return -self.swf(sol.c, sol.b, sol.ell)

        # g. define bounds so each variable is between 0 and 1
        bounds = [(0, 1), (0, 1), (0, None)]

        # e. additional equality constraint: government budget must balance
        cons = [
            {'type': 'eq', 'fun': lambda x: x[0]*phi*w*(1-workerProblem().worker(phi=phi, tau=x[0], w=w, pb=1, pc=1, l=x[2]).ell) + x[1]*z + x[2] - parGov.g}
        ]
        
        # f. initial guess (three variables: tau_w, tau_z, and l)
        x0 = [0.2, 0.1, 0.1]  # start within the feasible region
        
        # g. solve using a constrained optimizer
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
        
        # h. save solution
        solGov.tau_w = res.x[0]
        solGov.tau_z = res.x[1]
        solGov.l = res.x[2]
         # i. print solution
        print(f'solution: tau_w={solGov.tau_w:.2f}, tau_z={solGov.tau_z:.2f}, l={solGov.l:.2f}')
        
        # j. return solution
        return solGov
        
# test
model = govSWF()
model.government(w=15, phi=0.5, z=0.1)