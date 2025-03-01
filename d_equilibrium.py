from types import SimpleNamespace
from scipy.optimize import minimize, bisect
from a_hh import hhProblem
from b_firm import firmProblem

class equilibirium():
    
    def __init__(self):
        # a. define vector of equilivirum variables
        parEq = self.parEq=SimpleNamespace()

    def evaluate_equilibrium(self, w, p):
        parEq = self.parEq
        
        # a. optimal behavior of firm
        firm = firmProblem()
        firmSol = firm.firm(epsilon=0.5, w=w, tau_z=1, p=p)
        t = firmSol.t 
        y = firmSol.y

        # b. optimal behavior of hould
        hh = hhProblem()
        hhSol = hh.hh(phi=0.5, tau=0.2, w=w, pb=p, pc=p, l=0)
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

eq = equilibirium()
equilibrium_price = eq.evaluate_equilibrium(w=1, p=1)