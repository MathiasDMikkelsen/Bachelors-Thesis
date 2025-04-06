import sympy
from sympy import symbols, Eq, nsolve

# ------------------------------------------------------------------------------
# 1) Declare symbols
# ------------------------------------------------------------------------------
t = symbols('t', positive=True)       # labor (or "resource") choice
z = symbols('z', positive=True)       # pollution choice
p, w, tau_z, c, sigma = symbols('p w tau_z c sigma', positive=True)
x = symbols('x', positive=True)       # Constraint parameter: z <= x*t

# ------------------------------------------------------------------------------
# 2) Define the production function and r = (sigma-1)/sigma
# ------------------------------------------------------------------------------
def f(t_, z_, c_, sigma_):
    r_ = (sigma_ - 1) / sigma_
    return (c_ * t_**r_ + (1 - c_) * z_**r_)**(1/r_)

r_sym = (sigma - 1) / sigma
f_expr = f(t, z, c, sigma)

# ------------------------------------------------------------------------------
# 3) Write first-order conditions (interior solution)
# ------------------------------------------------------------------------------
eq1 = Eq(w, c * t**(r_sym - 1) * f_expr**(1 - r_sym) * p)        # FOC with respect to t
eq2 = Eq(tau_z, (1 - c) * z**(r_sym - 1) * f_expr**(1 - r_sym) * p)  # FOC with respect to z

# ------------------------------------------------------------------------------
# 4) Solve system numerically for (t, z)
# ------------------------------------------------------------------------------
# Example parameter values – using sigma = 2.0 yields r = 0.5.
param_vals = {
    p: 2.0,      # price of the good
    w: 1.0,      # wage
    tau_z: 0.5,  # pollution tax
    c: 0.3,      # factor share of labor
    sigma: 2.0,  # elasticity parameter (=> r = 0.5)
    x: 1.0       # constraint: z cannot exceed t (z <= t)
}

# Solve eq1 for t by substituting z = 1 (to get a univariate equation)
sol_t = nsolve(eq1.subs(param_vals).subs(z, 1), 0.9)

# Now solve eq2 for z by substituting the found value of t
sol_z = nsolve(eq2.subs(param_vals).subs(t, sol_t), 0.9)

print("Interior solution (unconstrained):")
print(" t* =", sol_t)
print(" z* =", sol_z)

# ------------------------------------------------------------------------------
# 5) Check the constraint: z <= x*t
#    If the solution violates z <= x*t, then set z = x*t and solve again for t.
# ------------------------------------------------------------------------------
if sol_z > param_vals[x] * sol_t:
    # Corner solution: enforce z = x*t in eq1
    eq_corner = eq1.subs(param_vals).subs(z, param_vals[x] * t)
    # Check whether eq_corner still depends on t
    if eq_corner.free_symbols:
        sol_t_corner = nsolve(eq_corner, 0.9)
    else:
        # If the equation is constant (no free symbols), use the interior solution.
        sol_t_corner = sol_t
    sol_z_corner = param_vals[x] * sol_t_corner
    print("\nCorner solution required (z = x*t).")
    print(" t* =", sol_t_corner)
    print(" z* =", sol_z_corner)
else:
    print("\nConstraint is satisfied by the interior solution.")