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

        # b. optimal behavior of household
        hh = workerProblem()
        hhSol = hh.worker(phi=1, tau=0, w=w, pb=p, pc=p, l=0.25*z+profit)
        h = 1 - hhSol.ell 
        c = hhSol.c
        b = hhSol.b
        
        # a. optimal behavior of firm
        firm = firmProblem()
        firmSol = firm.solve_firm(w=8, tau_z=5, y=c+b)
        t = firmSol.t 
        z = firmSol.z
        y = firmSol.y
        profit = firmSol.profit

        # c. excess demand
        g_m = (c+b)-y # goods market excess demand
        l_m = t-(1-hhSol.ell) # labor market excess demand
        
        # d. save parameters
        parEq.g_m = g_m
        parEq.l_m = l_m
        
        print(f'excess goods market demand: {parEq.g_m:.2f}, excess labor market demand:={parEq.l_m:.2f}')
        
        return parEq
    
eq = equilibirium()
eq.evaluate_equilibrium(1,1)