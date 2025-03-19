import math
from pyomo.environ import (
    ConcreteModel, Var, Param, RangeSet, Constraint, Expression, Objective,
    SolverFactory, exp, value
)
import pyomo.environ as pyo

# --------------------------------------------------
# Relaxed Calibration Settings
# --------------------------------------------------
n = 5                      # Number of households
T_total = 1.0              # Total time endowment per household
D0_val = 0.5             # Subsistence level for polluting consumption
G_val = 0.0                # Relaxed government spending
tau_z_val = 10.0         # Relaxed environmental tax on production

# --------------------------------------------------
# Create the model
# --------------------------------------------------
model = ConcreteModel()

# Sets
model.I = RangeSet(1, n)

# --------------------------------------------------
# Parameters
# --------------------------------------------------
model.alpha = Param(initialize=0.7)
model.beta  = Param(initialize=0.2)
model.gamma = Param(initialize=0.2)
model.d0    = Param(initialize=D0_val)         # D0 = 0.5
model.x     = Param(initialize=10.0)          # Technology parameter x
model.p_c   = Param(initialize=1.0)            # Clean good price (numeraire)
model.eps_c = Param(initialize=0.995)          # Clean firm labor share
model.eps_d = Param(initialize=0.92)           # Dirty firm labor share
model.r     = Param(initialize=-1)           # r = -1
model.tau_z = Param(initialize=tau_z_val)      # Environmental tax τ_z = 0.05
# Income tax rates on households (all zero here)
tau_w_data = {i: 0.0 for i in model.I}
model.tau_w = Param(model.I, initialize=tau_w_data)
# Productivity vector φ (as in the paper)
phi_data = {
    1: 0.2,
    2: 0.2,
    3: 0.2,
    4: 0.2,
    5: 0.2
}
model.phi = Param(model.I, initialize=phi_data)
model.t_total = Param(initialize=T_total)      # Total time endowment per household
model.G       = Param(initialize=G_val)          # Government spending (relaxed)

# --------------------------------------------------
# Decision Variables
# --------------------------------------------------
# Household-specific raw variables:
#   d_i = d0 + exp(d_raw_i)
#   ℓ_i = 1/(1 + exp(-ell_raw_i)) ∈ (0,1)
model.d_raw   = Var(model.I, domain=pyo.Reals)
model.ell_raw = Var(model.I, domain=pyo.Reals)
model.lam     = Var(model.I, domain=pyo.Reals)

# Firm and market variables (transformed from u15...u20):
#   t_c = T_total/(1+exp(-u15)) ∈ (0, T_total)
#   t_d = T_total/(1+exp(-u16)) ∈ (0, T_total)
#   z_c = exp(u17),   z_d = exp(u18)
#   p_d = exp(u19),   w = exp(u20)
model.u15 = Var(domain=pyo.Reals)
model.u16 = Var(domain=pyo.Reals)
model.u17 = Var(domain=pyo.Reals)
model.u18 = Var(domain=pyo.Reals)
model.u19 = Var(domain=pyo.Reals)
model.u20 = Var(domain=pyo.Reals)

# --------------------------------------------------
# Transformed Expressions
# --------------------------------------------------
def d_expr_rule(m, i):
    return m.d0 + exp(m.d_raw[i])
model.d = Expression(model.I, rule=d_expr_rule)

def ell_expr_rule(m, i):
    return 1/(1 + exp(-m.ell_raw[i]))
model.ell = Expression(model.I, rule=ell_expr_rule)

model.t_c = Expression(expr = model.t_total/(1 + exp(-model.u15)) )
model.t_d = Expression(expr = model.t_total/(1 + exp(-model.u16)) )
model.z_c = Expression(expr = exp(model.u17) )
model.z_d = Expression(expr = exp(model.u18) )
model.p_d = Expression(expr = exp(model.u19) )
model.w   = Expression(expr = exp(model.u20) )

# Lump-sum transfer (l_val)
def l_val_rule(m):
    tax_rev = sum(m.tau_w[i] * m.phi[i] * m.w * (m.t_total - m.ell[i]) for i in m.I) \
              + (m.z_c + m.z_d)*m.tau_z
    return (tax_rev - m.G) / n
model.l_val = Expression(rule=l_val_rule)

# --------------------------------------------------
# Household Conditions (First-Order Conditions)
# --------------------------------------------------
def hh_c_rule(m, i):
    c_i = (((1 - m.tau_w[i])*m.phi[i]*m.w*(m.t_total - m.ell[i]) + m.l_val)
           - m.p_d*m.d[i]) / m.p_c
    return m.alpha/c_i - m.lam[i]*m.p_c == 0
model.hh_c_constr = Constraint(model.I, rule=hh_c_rule)

def hh_d_rule(m, i):
    return m.beta/(m.d[i] - m.d0) - m.lam[i]*m.p_d == 0
model.hh_d_constr = Constraint(model.I, rule=hh_d_rule)

def hh_ell_rule(m, i):
    return m.gamma/m.ell[i] - m.lam[i]*(1 - m.tau_w[i])*m.phi[i]*m.w == 0
model.hh_ell_constr = Constraint(model.I, rule=hh_ell_rule)

def household_budget_constraint(m, i):
    c_i = (((1 - m.tau_w[i])*m.phi[i]*m.w*(m.t_total - m.ell[i]) + m.l_val)
           - m.p_d*m.d[i]) / m.p_c
    return c_i >= 1e-6
model.household_budget_cons = Constraint(model.I, rule=household_budget_constraint)

# (Here we remove the leisure upper-bound constraint to allow more flexibility.)

# --------------------------------------------------
# Firm Conditions: Production and FOCs
# --------------------------------------------------
def firm_c_production_rule(m):
    return (m.eps_c * m.t_c**m.r + (1 - m.eps_c) * m.z_c**m.r)**(1/m.r)
model.y_c = Expression(rule=firm_c_production_rule)

def firm_c_t_rule(m):
    dy_dt = m.eps_c * m.t_c**(m.r - 1) * model.y_c**(1 - m.r)
    return m.p_c * dy_dt - m.w == 0
model.firm_c_t_constr = Constraint(rule=firm_c_t_rule)

def firm_c_z_rule(m):
    dy_dz = (1 - m.eps_c) * m.z_c**(m.r - 1) * model.y_c**(1 - m.r)
    return m.p_c * dy_dz - m.tau_z == 0
model.firm_c_z_constr = Constraint(rule=firm_c_z_rule)

def firm_d_production_rule(m):
    return (m.eps_d * m.t_d**m.r + (1 - m.eps_d) * m.z_d**m.r)**(1/m.r)
model.y_d = Expression(rule=firm_d_production_rule)

def firm_d_t_rule(m):
    dy_dt = m.eps_d * m.t_d**(m.r - 1) * model.y_d**(1 - m.r)
    return m.p_d * dy_dt - m.w == 0
model.firm_d_t_constr = Constraint(rule=firm_d_t_rule)

def firm_d_z_rule(m):
    dy_dz = (1 - m.eps_d) * m.z_d**(m.r - 1) * model.y_d**(1 - m.r)
    return m.p_d * dy_dz - m.tau_z == 0
model.firm_d_z_constr = Constraint(rule=firm_d_z_rule)

# --------------------------------------------------
# Market Clearing Conditions
# --------------------------------------------------
def market_dirty_rule(m):
    return sum(m.d[i] for i in m.I) - (model.y_d - 0.5*model.G/model.p_d) == 0
model.market_dirty_constr = Constraint(rule=market_dirty_rule)

def market_labor_rule(m):
    return sum(m.phi[i]*(m.t_total - m.ell[i]) for i in m.I) - (model.t_c + model.t_d) == 0
model.market_labor_constr = Constraint(rule=market_labor_rule)

# --------------------------------------------------
# Additional Constraint: Clean Firm Production Bound
# --------------------------------------------------
model.clean_production_bound = Constraint(expr = model.z_c <= model.x * model.t_c)

# --------------------------------------------------
# Lower Bound on Clean Firm Labor Input (t_c ≥ 0.1)
# --------------------------------------------------
model.t_c_lb = Constraint(expr = model.t_c >= 0.1)

# --------------------------------------------------
# Dummy Objective (Feasibility Problem)
# --------------------------------------------------
model.obj = Objective(expr=0.0)

# --------------------------------------------------
# Initial Guesses
# --------------------------------------------------
for i in model.I:
    model.d_raw[i].value = -0.5    # d_i = 0.5 + exp(-0.5) ≈ 0.5 + 0.6065 = 1.1065
    model.ell_raw[i].value = 0.0   # ℓ_i = 0.5
    model.lam[i].value = 1.0

model.u15.value = 0.0    # t_c = 0.5 (will be >= 0.1)
model.u16.value = 0.0    # t_d = 0.5
model.u17.value = 0.0    # z_c = 1
model.u18.value = 0.0    # z_d = 1
model.u19.value = 0.0    # p_d = 1
model.u20.value = math.log(5)   # w = 5

# --------------------------------------------------
# Solve the Model
# --------------------------------------------------
solver = SolverFactory('ipopt')
results = solver.solve(model, tee=True)

# --------------------------------------------------
# Display Results
# --------------------------------------------------
print("Household values:")
for i in model.I:
    d_val = value(model.d[i])
    ell_val = value(model.ell[i])
    lam_val = value(model.lam[i])
    print(f"Household {i}: d = {d_val:.4f}, ℓ = {ell_val:.4f}, λ = {lam_val:.4f}")

print("\nFirm and market values:")
print(f"t_c = {value(model.t_c):.4f}")
print(f"t_d = {value(model.t_d):.4f}")
print(f"z_c = {value(model.z_c):.4f}")
print(f"z_d = {value(model.z_d):.4f}")
print(f"p_d = {value(model.p_d):.4f}")
print(f"w   = {value(model.w):.4f}")