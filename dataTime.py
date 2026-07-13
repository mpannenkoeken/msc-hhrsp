import pandas as pd
import numpy as np
import gurobipy as grb
from pathlib import Path

# figure out the folder we're currently in
currDir = Path(__file__).resolve().parent

# go find the nearby data files we need
carersFile = currDir.parent / "Home HealthCare Data" / "NW_Carers.csv"
clientsFile = currDir.parent / "Home HealthCare Data" / "NW_Clients.csv"
visitsFile = currDir.parent / "Home HealthCare Data" / "NW_CareVisits.csv"
travelFile = currDir.parent / "Home HealthCare Data" / "NW_TravelTimes.csv"

# import those files so we have something to actually work with

# the carer file ends in a ragged column
carersData = []
with open(carersFile) as f:
    # get the header row outta the way
    header = next(f).strip().split(";")
    
    # go row by row through this god-awful mess pretending to be a csv
    for line in f:
        # extract each value
        parts = [p.strip() for p in line.split(";")]
        # first five cols are consistent w/ csv structure
        fixedLine = parts[:5]
        # \exists a variable length tail in last col(s)
        tail = [p for p in parts[5:] if p != ""]
        
        # make a fixed row and add to running data list
        row = fixedLine + [list(map(str, tail)) if tail else []]
        carersData.append(row)

carers_df = pd.DataFrame(carersData, columns=header, index=None)
# print(carers_df.head)

# the client file ends in a ragged column
clientsData = []
with open(clientsFile) as f:
    # get the header row outta the way
    newheader = next(f).strip().split(";")
    
    # go row by row through this god-awful mess pretending to be a csv
    for line in f:
        # extract each value
        parts = [p.strip() for p in line.split(";")]
        # first two cols are consistent w/ csv structure
        fixedLine = parts[:2]
        # \exists a variable length tail in last col(s)
        tail = [p for p in parts[2:] if p != ""]
        
        # make a fixed row and add to running data list
        row = fixedLine + [list(map(int, tail)) if tail else []]
        clientsData.append(row)
        
clients_df = pd.DataFrame(clientsData, columns=["Client ID", "Gender Preference", "Known Carers"], index=None)
# print(clients_df.head)

# these files should be relatively straightforward
visits_df = pd.read_csv(visitsFile, sep=";", index_col="Visit ID")
# print(visits_df.head)
travel_df = pd.read_csv(travelFile, sep=";", index_col=0)
travel_df = travel_df.dropna(axis=1, how="all")
# print(travel_df.head)

# TO-DO: Get the Easy Parameters into np.arrays

# TO-DO: Calculate Localities by Multidimensional Scaling

# TO-DO: Build and Solve the Model