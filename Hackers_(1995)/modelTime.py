#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:37:47 2026

@author: nolanbearw
"""

import gurobipy as grb
import pickle
import numpy as np
import pandas as pd
import os
from pathlib import Path
import math

"""
INITIALIZATIONS
"""
# figure out the folder we're currently in
currDir = Path(__file__).resolve().parent
currParent = currDir.parent
currAbuela = currParent.parent

# go find the nearby input pickle we need
carersFile = currAbuela / "Home HealthCare Data" / "inputs.pkl"

# get the pickled sets and parameters out
with open(carersFile, "rb") as f:
    data = pickle.load(f)

D = data["D"]
Cd = data["Cd"]
CdD = data["CdD"]
L = data["L"]
dij = data["dij"]
ril = data["ril"]
Fijd = data["Fijd"]
fijd = data["fijd"]
sid = data["sid"]
Vd = data["Vd"]
Vlpd = data["Vlpd"]
Vlsd = data["Vlsd"]
li = data["li"]
K = data["K"]

# define the tolerances on geospatial solo/pair distributions
epsi = 2 # tolerance on solo caregiving units
delta = 1 # tolerance on pair caregiving units

# do the set minus to get the non-drivers now instead of each time
Cd_minus_CdD = {d: set(Cd[d]) - set(CdD[d]) for d in D}

# Initialize the model
m = grb.Model("Carer_Assignments")

"""
DECLARATION OF DECISION VARIABLES
"""
# represents the linearization of $z_{il}^d y_{i}^d$
vild = m.addVars(((i,l,d) for d in D for i in CdD[d] for l in L), 
                 vtype=grb.GRB.CONTINUOUS)
# represents the linearization of $z_{il}^d x_{ij}^d$
wijld = m.addVars(((i,j,l,d) for d in D for i in CdD[d] for j in Cd[d] if j != i for l in L), 
                  vtype=grb.GRB.CONTINUOUS)

# 1 if i \in C_D^d paired with j \in C^d \setminus \{i\} on day d, 0 else
xijd = m.addVars(((i,j,d) for d in D for i in CdD[d] for j in Cd[d] if j != i), 
                 vtype=grb.GRB.BINARY)
# 1 if caregiver j \in C^d is unpaired (i.e. does solo visits) on day d, 0 else
yid = m.addVars(((i,d) for d in D for i in Cd[d]), vtype=grb.GRB.BINARY)
# 1 if caregiver i \in C_D^d assigned to locality l \in L on dayd, 0 else
zild = m.addVars(((i,l,d) for d in D for i in CdD[d] for l in L), 
                 vtype=grb.GRB.BINARY)

"""
OBJECTIVES
"""
# objective one is to maximize the potential of familiar visits
potentialFamiliarVisits = grb.quicksum(
                            grb.quicksum(sid.get((j,d),0) * yid[j,d] + 
                                         grb.quicksum(Fijd.get((i,j,d),0) * xijd[i,j,d]
                                                      + fijd.get((i,j,d),0) * xijd[i,j,d]
                                                      for i in CdD[d] if i != j)
                                         for j in Cd[d]) 
                            for d in D)
# objective two is to minimize the travel time for caregivers' assignments
carerTravelCeiling = grb.quicksum(
                         grb.quicksum(
                             grb.quicksum(dij[i].loc[j] * xijd[i,j,d] +
                                    grb.quicksum(ril[(i,l)] * zild[i,l,d]
                                        for l in L)
                                 for i in CdD[d] if i != j)
                             for j in Cd[d])
                        for d in D)
"""
CONSTRAINTS
"""
# assign each available caregiver once per day available (non-driver)
m.addConstrs(yid[j,d] + grb.quicksum(xijd[i,j,d] for i in CdD[d] if i != j) == 1
             for d in D for j in Cd_minus_CdD[d])

# assign each available caregiver once per day available (driver)
m.addConstrs(yid[i,d] + 
             grb.quicksum(xijd[i,j,d] for j in Cd[d] if j != i) +
             grb.quicksum(xijd[j,i,d] for j in CdD[d] if j != i) == 1
             for d in D for i in CdD[d])
        
# do not pair caregivers greater than K apart
for d in D:
    for i in CdD[d]:
        for j in Cd[d]:
            if dij[i].loc[j] > K:        
                xijd[i,j,d].ub = 0
                
# upper bound the solo units and their geospatial distribution
m.addConstrs(grb.quicksum(li[(i,l)] * yid[i,d] for i in Cd_minus_CdD[d]) +
             grb.quicksum(vild[i,l,d] for i in CdD[d]) <= 
             math.ceil(Vlsd[(l,d)] / Vd[d] * len(Cd[d]))
             for l in L for d in D)

# lower bound the solo units and their geospatial distribution
m.addConstrs(grb.quicksum(li[(i,l)] * yid[i,d] for i in Cd_minus_CdD[d]) +
             grb.quicksum(vild[i,l,d] for i in CdD[d]) >= 
             math.floor(Vlsd[(l,d)] / Vd[d] * len(Cd[d]) / 2)
             for l in L for d in D)

print(Vlpd.keys())

# upper bound the pair units and their geospatial distribution
m.addConstrs(grb.quicksum(
                grb.quicksum(wijld[i,j,l,d] for j in Cd[d] if j != i)
                for i in CdD[d]) <= math.ceil(Vlpd.get((l,d), 0) / Vd[d] * len(Cd[d]))
            for l in L for d in D)

# lower bound the pair units and their geospatial distribution
m.addConstrs(grb.quicksum(
                grb.quicksum(wijld[i,j,l,d] for j in Cd[d] if j != i)
                for i in CdD[d]) <= math.floor(Vlpd.get((l,d),0) / Vd[d] * len(Cd[d]) / 2)
            for l in L for d in D)

# make wijld do what i want it to
m.addConstrs(wijld[i,j,l,d] <= xijd[i,j,d] for d in D for i in CdD[d] for j in Cd[d] if j != i for l in L)
m.addConstrs(wijld[i,j,l,d] <= zild[j,l,d] for d in D for i in CdD[d] for j in Cd[d] if j != i for l in L)
m.addConstrs(wijld[i,j,l,d] >= zild[j,l,d] + xijd[i,j,l,d] - 1
             for d in D for i in CdD[d] for j in Cd[d] if j != i for l in L)

# make vild do what i want it to
m.addConstrs(vild[i,l,d] <= yid[i,d] for d in D for i in CdD[d] for l in L)
m.addConstrs(vild[i,l,d] <= zild[i,l,d] for d in D for i in CdD[d] for l in L)
m.addConstrs(vild[i,l,d] >= yid[i,d] + zild[i,l,d] - 1
             for d in D for i in CdD[d] for l in L)

"""
GUROBI WORKS ITS MAGIC
"""

"""
EXPORT RESULTS
"""