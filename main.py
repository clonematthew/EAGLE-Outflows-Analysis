# Imports of python libraries
import numpy as np
import pandas as pd
import csv
import os.path

# Imports of my libraries
from sqlFunctions import *
from galaxyClass import galaxy
from geometricFunctions import getCylinders

# Setting the path to the datafiles
filepath = "/home/universe/spxtd1-shared/RefL0100N1504/snapshot_028_z000p000/snap_028_z000p000.0.hdf5"
#filepath = "./data/snap_028_z000p000.0.hdf5"

# Defining the simulation we want
sim = "REFL0100N1504"
#sim = "REFL0012N0188"
simFile = sim + "galaxyData.txt"

# Checking if that file exists
if os.path.exists(simFile):
    pass
else:
    getGalaxyData(sim)

print("Galaxy Data Loaded")

# Importing the galaxy data
dataFrame = pd.read_csv(simFile, names=["x", "y", "z", "gn", "sgn", "mass", "vx", "vy", "vz", "sfr", "id"])
dataFrame["id"] = dataFrame["id"].astype("int")

print("DataFrame Imported")

# Getting the cylinders to use
nCyls = 49
cylSize = 3 * 1e3 * 3.09e18
vectors, inclinations, angles, thetas = getCylinders(nCyls)

# Choosing sample of galaxies
sample = dataFrame[dataFrame["mass"] > 3e9]
nGals = len(sample["mass"])

# Opening csv writer and numpy writer
f = open("galaxyData.csv", "w")
writer = csv.writer(f)

print("Galaxy Output File Initialised")

# Header rows
headerRow = ["GalaxyID", "Mass", "Star Formation Rate", "Particles", "Velocity", "Theta", "Phi", "Inclination", "N Particles"]
writer.writerow(headerRow)

# Initialising i and v data
i = 0
vData = np.array([])

# Main loop
while i< nGals:
    # Choosing the galaxy
    d = sample.iloc[i]

    # Getting the galaxy data from the dataFrame row
    centre = np.array([d[0], d[1], d[2]])
    velocity = np.array([d[6], d[7], d[8]])
    gn = d[3]
    sgn = d[4]
    mass = int(d["mass"])
    sfr = d["sfr"]
    gid = d["id"]

    # Loading in the galaxy object
    gal = galaxy(gn, sgn, centre, velocity, 0.5, False, filepath)

    if gal.particles:

        for j in range(len(vectors)):
            # Initialising the cylinder
            v, w, p = gal.cylinder(vectors[j], cylSize)

            # Outputting the data
            rowList = [gid, mass, sfr, "True"]
            rowList = np.concatenate((rowList, [w]))
            endList = [thetas[j], angles[j], inclinations[j], p]
            rowList = np.concatenate((rowList, endList))
            writer.writerow(rowList)

            # Checking for particles being present
            if p == 0:
                # Writing null spectrum data
                vData = np.append(vData, "Null:")
            else:
                # Writting spectrum data
                vstr = v.tolist()
                values = str(vstr) + ":"
                vData = np.append(vData, values)

    else:
        # Priting a null array if no particles
        rowList = [gid, mass, sfr, "False", "Null", "Null", "Null", "Null", "Null"]
        writer.writerow(rowList)

        # Writing spectrum data
        vData = np.append(vData, "Null:")

    # Updating i
    i = i + 1

    if i % 50 == 0:
        print("Galaxy %s loaded"% i)

# Saving the spectra data
np.savetxt("spectra.txt", vData, fmt="%s")
    
print("Done")