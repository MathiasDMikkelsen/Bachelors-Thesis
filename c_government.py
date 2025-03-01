from types import SimpleNamespace
from scipy.optimize import minimize
import numpy as np
from scipy import optimize

class govSWF():
    def __init__(self):
        """ setup"""
        
        # a. define vector of parameters
        parGov = self.parGov = SimpleNamespace()
        
        # b. exogenous parameters
        parGov.o = 0.3
        
        # c. define solution set
        solGov = self.solGov = SimpleNamespace()
        
        # d. solution parameters
        solGov.t = 0 # clean input
        solGov.z = 0 # polluting input
      
    def production(self,t,z):
        """ production of firm """
        
        # a. retrieve exogenous parameters
        parFirm = self.parFirm
        
        # b. return production function
        return (parFirm.epsilon*t**parFirm.r + (1-parFirm.epsilon)*z**parFirm.r)**(1/parFirm.r)*(1 if z<=parFirm.x*t else 0)
    
    def firm(self, epsilon, w, tau_z, p):
        """ maximize profit for firms """
        
        # a. retrieve solution set and parameter vector
        solFirm = self.solFirm
        parFirm = self.parFirm
        
        # b. add endogenous parameters to parameter vector
        parFirm.w = w # wage
        parFirm.p = p  # price of output
        
        # c. add variable parameters to parameter vector
        parFirm.epsilon = epsilon  # share of polluting input in production
        parFirm.tau_z = tau_z      # pollution tax rate
        
        # d. define objective function as negative profit
        def obj(x):
            t, z = x # define variables
            return -p*self.production(t, z)+ w*t + tau_z*z
        
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