#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 13:37:47 2026

@author: nolanbearw
"""

import gurobipy as grb
import pickle
import pandas as pd
from pathlib import Path
import math

"""
HELPER FUNCTIONS
"""
def normalize(df):
    df = df.sort_index(axis=1)
    return df.sort_values(
        by=list(df.columns), kind="mergesort"
                ).reset_index(drop=True)

def get_sol(D, Cd, CdD, L, xijd, yid, zild, li):
    rows = []
    for d in D:
        # paired caregivers
        for i in CdD[d]:
            for j in Cd[d]:
                if j != i:
                    if xijd[i,j,d].X > 0.5:
                        # find locality assignment
                        lAssignment = next(
                            (l for l in L if zild[i,l,d].X > 0.5),
                            None
                            )
                        
                        if lAssignment == None:
                            lAssignment = next(l for l in L if li[(i,l)] == 1)
                        
                        rows.append({
                            "Day": d,
                            "Caregiver ID": i,
                            "Partner ID (if any)": int(j),
                            "Locale Assignment": lAssignment
                            })
                    
        # solo caregivers
        for i in Cd[d]:
            if yid[i,d].X > 0.5:
                if i in CdD[d]:
                    lAssignment = next(
                        (l for l in L if zild[i,l,d].X > 0.5),
                        None
                        )
                    if lAssignment == None:
                        lAssignment = next(l for l in L if li[(i,l)] == 1)
                    
                    rows.append({
                        "Day": d,
                        "Caregiver ID": i,
                        "Partner ID (if any)": None,
                        "Locale Assignment": lAssignment
                        })
                else:
                    lAssignment = next(l for l in L if li[(i,l)] == 1)
                    rows.append({"Day": d, "Caregiver ID": i, 
                                 "Partner ID (if any)": None, 
                                 "Locale Assignment": lAssignment})
    # save results
    currSol = pd.DataFrame(rows)
    return currSol
    
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
Vpd = data["Vpd"]
Vsd = data["Vsd"]
Vlpd = data["Vlpd"]
Vlsd = data["Vlsd"]
Pd = data["Pd"]
Sd = data["Sd"]
li = data["li"]
K = data["K"]

# define the tolerances on geospatial solo/pair distributions
epsi = 0 # tolerance on solo caregiving units
delta = 0 # tolerance on pair caregiving units

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
wijld = m.addVars(((i,j,l,d) for d in D for i in CdD[d] 
                   for j in Cd[d] if j != i for l in L), 
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
                                         grb.quicksum(Fijd.get((i,j,d),0) 
                                                      * xijd[i,j,d]
                                                      + fijd.get((i,j,d),0) 
                                                      * xijd[i,j,d]
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
m.addConstrs(yid[j,d] + grb.quicksum(xijd[i,j,d] for i in CdD[d] if i != j)
             == 1 for d in D for j in Cd_minus_CdD[d])

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
             math.ceil(Vlsd.get((l,d), 0) / Vsd[d] * (Sd[d] + epsi))
             for l in L for d in D)

# lower bound the solo units and their geospatial distribution
m.addConstrs(grb.quicksum(li[(i,l)] * yid[i,d] for i in Cd_minus_CdD[d]) +
             grb.quicksum(vild[i,l,d] for i in CdD[d]) >= 
             math.floor(Vlsd.get((l,d), 0) / Vsd[d] * (Sd[d] - epsi))
             for l in L for d in D)

# upper bound the pair units and their geospatial distribution
m.addConstrs(grb.quicksum(
                grb.quicksum(wijld[i,j,l,d] for j in Cd[d] if j != i)
                for i in CdD[d]) 
                <= math.ceil(Vlpd.get((l,d), 0) / Vpd[d] * (Pd[d] + delta))
            for l in L for d in D)

# lower bound the pair units and their geospatial distribution
m.addConstrs(grb.quicksum(
                grb.quicksum(wijld[i,j,l,d] for j in Cd[d] if j != i)
                for i in CdD[d]) 
                <= math.floor(Vlpd.get((l,d), 0) / Vpd[d] * (Pd[d] + delta))
            for l in L for d in D)

# get total pairs within tolerance
m.addConstrs(grb.quicksum(
                grb.quicksum(
                    xijd[i,j,d] for j in Cd[d] if j != i)
                for i in CdD[d]) <= Pd[d] + delta
    for d in D)

m.addConstrs(grb.quicksum(
                grb.quicksum(
                    xijd[i,j,d] for j in Cd[d] if j != i)
                for i in CdD[d]) >= Pd[d] - delta
    for d in D)

# get total solos within tolerance
m.addConstrs(
                grb.quicksum(
                    yid[i,d] for i in Cd[d])
                <= Sd[d] + epsi
    for d in D)

m.addConstrs(
                grb.quicksum(
                    yid[i,d] for i in Cd[d])
                >= Sd[d] - epsi
    for d in D)

# make wijld do what i want it to
m.addConstrs(wijld[i,j,l,d] <= xijd[i,j,d] for d in D for i in CdD[d] 
             for j in Cd[d] if j != i for l in L)
m.addConstrs(wijld[i,j,l,d] <= zild[i,l,d] for d in D for i in CdD[d] 
             for j in Cd[d] if j != i for l in L)
m.addConstrs(wijld[i,j,l,d] >= zild[i,l,d] + xijd[i,j,d] - 1
             for d in D for i in CdD[d] for j in Cd[d] if j != i for l in L)

# make vild do what i want it to
m.addConstrs(vild[i,l,d] <= yid[i,d] for d in D for i in CdD[d] for l in L)
m.addConstrs(vild[i,l,d] <= zild[i,l,d] for d in D for i in CdD[d] for l in L)
m.addConstrs(vild[i,l,d] >= yid[i,d] + zild[i,l,d] - 1
             for d in D for i in CdD[d] for l in L)

"""
GUROBI WORKS ITS MAGIC
"""
# optimize for familiarity/objective 1
m.setObjective(potentialFamiliarVisits, sense=grb.GRB.MAXIMIZE)
m.Params.OutputFlag = 0
m.optimize()
# store travel time in unrestricted case
maxTravel = carerTravelCeiling.getValue()

allResults = []
# store objective 1 results
maxTravelSol = get_sol(D, Cd, CdD, L, xijd, yid, zild, li)
allResults.append(maxTravelSol)

print(f"Maximum Observed Travel Ceiling: {maxTravel}")

# add constraint on travel time
travel = m.addConstr(carerTravelCeiling <= maxTravel)

# optimize for travel time/objective 2
m.setObjective(carerTravelCeiling, sense=grb.GRB.MINIMIZE)
m.update
m.optimize()
# store travel time when minimized
minTravel = m.ObjVal

# store objective 2 results
minTravelSol = get_sol(D, Cd, CdD, L, xijd, yid, zild, li)
allResults.append(minTravelSol)

print(f"Minimum Observed Travel Ceiling: {minTravel}")

# reset model objective as objective 1/familiarity
m.setObjective(potentialFamiliarVisits, sense=grb.GRB.MAXIMIZE)
m.update()

# optimize for linear combinations of min and max travel times
for lamb in [0.25, 0.5, 0.75]:
    # update rhs of travel constraint and solve
    travel.rhs = lamb * maxTravel + (1 - lamb) * minTravel
    m.update()
    print(f"Current Travel RHS: {travel.rhs}")
    m.optimize()
    print(f"Observed Travel: {carerTravelCeiling.getValue()}")
    
    # save results
    currSol = get_sol(D, Cd, CdD, L, xijd, yid, zild, li)
    allResults.append(currSol)
    
"""
COMPARE RESULTS AND EXPORT WHILE IGNORING DUPLICATES
"""
unique_results = []
seen = []
for df in allResults:
    norm=normalize(df)
    if not any(norm.equals(existing) for existing in seen):
        seen.append(norm)
        unique_results.append(df)


for i, df in enumerate(unique_results):
    unique_results[i].to_csv(f"candidate_pairings_{i+1}.csv", index=False)
    