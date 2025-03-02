from types import SimpleNamespace
from scipy.optimize import bisect
from a_hh import workerProblem
import numpy as np
from b_firm import firmProblem
from scipy import optimize

class equilibirium():
    
    def __init__(self):
        # a. define vector of equilivirum variables
        parEq = self.parEq=SimpleNamespace()
        solEq = self.solEq=SimpleNamespace()
        
    def evaluate_equilibrium(self, p, w):
        parEq = self.parEq
        
        # a. optimal behavior of firm
        firm = firmProblem()
        firmSol = firm.solve_firm(p=p, tau_z=0.25 ,w=w)
        t = firmSol.t 
        y = firmSol.y

        # b. optimal behavior of household
        hh = workerProblem()
        hhSol = hh.worker(phi=1, tau=0, w=w, pb=p, pc=p, l=0)
        h = 1- hhSol.ell 
        c = hhSol.c

        # c. excess demand
        g_m = c-y # goods market excess demand
        l_m = t-h # labor market excess demand
        
        # d. save parameters
        parEq.g_m = g_m
        parEq.l_m = l_m
        
        print(f'excess goods market demand: {parEq.g_m:.2f}, excess labor market demand:={parEq.l_m:.2f}')
        
        return parEq
    
    def find_equilibrium(self):

        parEq = self.parEq
        solEq = self.solEq   

        # c. bisection search
        def obj(x):
            p,w = x
            self.evaluate_equilibrium(p,w)
            return parEq.g_m**2 + parEq.l_m**2

        res = optimize.minimize(obj,x0=[0.5, 0.5], method='L-BFGS-B', options={'ftol': 1e-12, 'gtol': 1e-12})
        solEq.p = res.x[0]
        solEq.w = res.x[1]
        print(f'the equilibrium wage is {solEq.w:.4f}')
        print(f'the equilibrium price is {solEq.p:.4f}')

eq = equilibirium()
eq.find_equilibrium()
eq.evaluate_equilibrium(eq.solEq.w, eq.solEq.p)