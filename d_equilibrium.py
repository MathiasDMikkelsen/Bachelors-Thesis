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
        hhSol = hh.worker(phi=1, tau=0, w=w, pb=p, pc=p, l=0.25*firmSol.z)
        h = 1 - hhSol.ell 
        c = hhSol.c

        # c. excess demand
        g_m = c-y # goods market excess demand
        l_m = t-h # labor market excess demand
        
        # d. save parameters
        parEq.g_m = g_m
        parEq.l_m = l_m
        
        print(f'excess goods market demand: {parEq.g_m:.2f}, excess labor market demand:={parEq.l_m:.2f}')
        
        return parEq
    
    def find_equilibrium_iterative(self):
        """
        Iteratively update the wage and price.
        1. For a fixed price, find the wage that zeroes the goods market excess demand.
        2. Holding that wage fixed, find the price that zeroes the labor market excess demand.
        3. Repeat until the price converges.
        """
        tol = 1e-6
        max_iter = 50

        # initial guess for price
        p_guess = 0.5

        for i in range(max_iter):
            # Step 1: With current p_guess, find wage w_sol so that goods market clears.
            def wage_obj(w):
                # evaluate equilibrium for given (p_guess, w)
                self.evaluate_equilibrium(p_guess, w)
                return self.parEq.g_m  # we want g_m = 0
            
            try:
                wage_sol = optimize.root_scalar(wage_obj, bracket=[0.1, 10.0],
                                                method='bisect', xtol=tol).root
            except Exception as e:
                print(f"Error in wage root finding: {e}")
                break

            # Step 2: With wage fixed, update price by clearing the labor market.
            def price_obj(p):
                self.evaluate_equilibrium(p, wage_sol)
                return self.parEq.l_m  # we want l_m = 0
            
            try:
                p_new = optimize.root_scalar(price_obj, bracket=[0.1, 10.0],
                                            method='bisect', xtol=tol).root
            except Exception as e:
                print(f"Error in price root finding: {e}")
                break

            print(f"Iteration {i+1}: p_old = {p_guess:.6f}, p_new = {p_new:.6f}, wage = {wage_sol:.6f}")
            if abs(p_new - p_guess) < tol:
                p_guess = p_new
                break
            
            p_guess = p_new

        self.solEq.p = p_guess
        self.solEq.w = wage_sol
        print(f"\nEquilibrium found: price = {self.solEq.p:.4f}, wage = {self.solEq.w:.4f}")
        # Optionally, evaluate the final equilibrium to print excess demands:
        self.evaluate_equilibrium(self.solEq.p, self.solEq.w)
        return self.solEq

eq = equilibirium()
eq.find_equilibrium_iterative()
eq.evaluate_equilibrium(eq.solEq.w, eq.solEq.p)