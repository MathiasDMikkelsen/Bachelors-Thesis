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
        self.parGov.omega = 0.3   # inequality aversion parameter
        self.parGov.g = 1         # government spending requirement
        self.parGov.time = 1      # total time endowment
        
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
    
    def government(self, c, b, ell, w, phi, z)
        """ maximize welfare """
        
        # a. retrieve solution set and parameter vector
        solGov = self.solGov
        parGov = self.parGov
        
        # b. add endogenous parameters to parameter vector
        parGov.c = c # optimal consumption of good c
        parGov.b = b  # optimal consumption of good b
        parGov.ell = ell  # optimal leisure supply
        parGov.w = w  # equilibrium wage
        parGov.z = z # optimal pollution
        
        # c. add variable parameters to parameter vector
        parGov.phi = phi  # share of polluting input in production
        
        # d. define objective function as negative welfare with budget constraint inserted
        def obj(x):
            tau_w, tau_z, l = x # define variables
            return -self.swf(c, b, parGov.time-(parGov.g+l-tau_z*z)/(tau_w*phi*w))
        
        # e. constraints
        cons = [ 
            {'type': 'ineq', 'fun': lambda x: x[0]},  # t >= 0
            {'type': 'ineq', 'fun': lambda x: x[1]},  # z >= 0
        ]
        
        # f. initial guess
        x0 = [1,1]  # start within feasible region 
        
        # g. solve using a constrained optimizer
        res = minimize(obj, x0, method='SLSQP', constraints=cons)
        
        # h. save solution
        solFirm.t = res.x[0]
        solFirm.z = res.x[1]
        solFirm.y = self.production(solFirm.t, solFirm.z)
        solFirm.pi = p*self.production(solFirm.t, solFirm.z)- w*solFirm.t - tau_z*solFirm.z
        
         # i. print solution
        print(f'solution: t={solFirm.t:.2f}, z={solFirm.z:.2f}, y={solFirm.y:.2f}, profit: {solFirm.pi:2f}')
        
        # j. return solution
        return solFirm
        
# test
model = firmProblem()
model.firm(epsilon=0.5, w=1, tau_z=0.2, p=1)