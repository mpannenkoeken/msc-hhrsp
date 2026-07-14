import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import pickle

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
clients_df["Client ID"] = clients_df["Client ID"].astype(int)
# these files should be relatively straightforward
visits_df = pd.read_csv(visitsFile, sep=";", index_col="Visit ID")

string_dur = visits_df["Visit Duration"].astype(str).str.strip()

# split into hours and minutes
hm = string_dur.str.split(":", expand=True)

visits_df["Visit Duration"] = (
    pd.to_timedelta(hm[0].astype(int), unit="h") +
    pd.to_timedelta(hm[1].astype(int), unit="m")
)

visits_df = visits_df.rename(columns={"ClientID": "Client ID"})

pair_visits_df = visits_df[visits_df["Number of Carers"] == 2]
single_visits_df = visits_df[visits_df["Number of Carers"] == 1]

pair_visits_df = pair_visits_df.merge(clients_df, on="Client ID")
single_visits_df = single_visits_df.merge(clients_df, on = "Client ID")

# travel time matrix ends up with an empty column
travel_df = pd.read_csv(travelFile, sep=";", index_col=0)
travel_df = travel_df.dropna(axis=1, how="all")

# Locality-Independent Parameters
# get the days we are planning
D = np.array(visits_df["Visit Date"].unique())

# figure out which carers and driving carers we have on those days
carers_exploded = carers_df.explode("Available Working Days")
drivers_exploded = carers_exploded.loc[carers_exploded["Driver"].astype(int) == 1]

Cd = (
    carers_exploded.groupby("Available Working Days")["Carer ID"]
    .apply(list)
    .to_dict()
)

CdD = (
    drivers_exploded.groupby("Available Working Days")["Carer ID"]
    .apply(list)
    .to_dict()
)

Vd = visits_df.groupby("Visit Date")["Visit Duration"].sum().to_dict()

K = pd.to_timedelta(40, unit="m")

explode_singles = single_visits_df.explode("Known Carers")

sid = explode_singles.groupby(["Known Carers", "Visit Date"])["Visit Duration"].sum().to_dict()

Fijd = defaultdict(lambda: pd.Timedelta(0))
fijd = defaultdict(lambda: pd.Timedelta(0))

pair_visits_df.columns = pair_visits_df.columns.str.strip().str.replace(" ", "_").str.lower()

for row in pair_visits_df.itertuples(index=False):
    d = getattr(row, "visit_date")
    duration = getattr(row, "visit_duration")
    known = set(getattr(row, "known_carers"))
    
    Cd_set = set(Cd[d])
    CdD_set = set(CdD[d])
    
    known = set(map(int, known))
    Cd_set = set(map(int, Cd[d]))
    CdD_set = set(map(int, CdD[d]))
    
    known_in_Cd = known & Cd_set
    known_in_CdD = known & CdD_set
    
    # --- Fijd: both i AND j known ---
    for i in known_in_CdD:
        for j in known_in_Cd:
            if i != j and j in known:
                Fijd[(i, j, d)] += duration
    
    # --- fijd: XOR ---
    for i in CdD_set:
        for j in Cd_set:
            if (i in known) ^ (j in known):
                fijd[(i, j, d)] += duration
                

# TO-DO: Calculate Localities by Multidimensional Scaling

# TO-DO: Build and Solve the Model