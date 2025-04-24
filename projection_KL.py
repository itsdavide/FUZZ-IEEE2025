#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Approximation code for the paper:

S. Lorenzini, A. Cinfrignini, D. Petturiti, and B. Vantaggi.
Quantile-constrained Choquet-Wasserstein p-box 
approximation of arbitrary belief functions.
In Proceedings of FUZZ-IEEE 2025.
"""

import pyomo.environ as pyo
from pyomo.environ import log
import numpy as np

optimizer_path = 'PATH_TO_BONMIN'

tolerance = 0.0000001

def projection_KL(q, J1, delta1, J2, delta2): 
    tot = sum(q)
    q = q / tot
    I = list(range(len(q)))
    # Create a model in  pyomo
    model = pyo.ConcreteModel()
    
    # Define the index set in pyomo
    model.I = pyo.Set(initialize=I)
    model.J1 = pyo.Set(initialize=J1)
    model.J2 = pyo.Set(initialize=J2)
    
    # Defines the constants in pyomo
    model.q = pyo.Param(model.I, initialize=q)
    model.delta1 = pyo.Param(initialize=delta1)
    model.delta2 = pyo.Param(initialize=delta2)
    
    # Defines the variables in pyomo
    model.p = pyo.Var(model.I, within=pyo.NonNegativeReals, bounds=(tolerance, np.Infinity), initialize=tolerance)
    model.KL = pyo.Var(bounds=(-np.Infinity, np.Infinity))
    
    # Set the constraints
    model.c1 = pyo.Constraint(expr=sum(model.p[i] * log(model.p[i] / model.q[i]) - model.p[i] + model.q[i] for i in model.I) == model.KL)
    model.c2 = pyo.Constraint(expr=sum(model.p[j] for j in model.J1) >= model.delta1)
    model.c3 = pyo.Constraint(expr=sum(model.p[j] for j in model.J2) >= model.delta2)
    model.c4 = pyo.Constraint(expr=sum(model.p[i] for i in model.I) == 1)
    
    model.o = pyo.Objective(expr = model.KL, sense=pyo.minimize)
    
    status = pyo.SolverFactory(optimizer_path).solve(model)
    pyo.assert_optimal_termination(status)

    
    opt_p = []
    for i in model.I:
        opt_p.append(pyo.value(model.p[i]))
    
    return (np.array(opt_p),  pyo.value(model.o))