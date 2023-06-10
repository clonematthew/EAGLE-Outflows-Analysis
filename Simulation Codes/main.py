# Imports of python libraries
import numpy as np
import pandas as pd
import csv
import os.path
from tqdm import tqdm

# Imports of my libraries
from sqlFunctions import *
from galaxyClass import galaxy
from geometricFunctions import getCylinders

# Setting the path to the datafiles
#prePath = "C:/Users/Work Account/OneDrive/Documents/University [MPhys]/Year 4/PX4310 - Physics Project/Code/"
#filepath = "/home/universe/spxtd1-shared/RefL0100N1504/snapshot_027_z010p000/snap_027_z010p000.0.hdf5"
#filepath = prePath+"/data/snap_028_z000p000.0.hdf5"
#filepath = "/home/universe/spxtd1-shared/NoAGNL0050N0752/snapshot_028_z000p000/snap_028_z000p000.0.hdf5"
filepath = "/home/universe/spxtd1-shared/NoAGNL0050N0752/snapshot_027_z000p101/snap_027_z000p101.0.hdf5"

# Defining the simulation we want

#sim = "REFL0100N1504"
#sim = "REFL0012N0188"
sim = "NoAGNL0050N0752"
simFile = sim + "galaxyData.txt"

# Checking if that file exists
if os.path.exists(simFile):
    pass
else:
    getGalaxyData(sim)

print("Galaxy Data Loaded")

# Importing the galaxy data
dataFrame = pd.read_csv(simFile, names=["x", "y", "z", "gn", "sgn", "mass", "vx", "vy", "vz", "sfr", "id", "rg", "rs", "r"])
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
f = open("galaxyData2.csv", "w")
writer = csv.writer(f)
fv = open("velocities.txt", "a")
fm = open("masses.txt", "a")
fs = open("sfrs.txt", "a")
ft = open("temperatures.txt", "a")

print("Galaxy Output File Initialised")

# Header rows
headerRow = ["GalaxyID", "Mass", "Star Formation Rate", "Particles", "Velocity", "Theta", "Phi", "Inclination", "N Particles"]
writer.writerow(headerRow)

# Initialising i and v data
i = 0

# Main loop
for i in tqdm(range(nGals)):
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
            v, w, p, m, s, t = gal.cylinder(vectors[j], cylSize)

            # Outputting the data
            rowList = [gid, mass, sfr, "True"]
            rowList = np.concatenate((rowList, [w]))
            endList = [thetas[j], angles[j], inclinations[j], p]
            rowList = np.concatenate((rowList, endList))
            writer.writerow(rowList)

            # Checking for particles being present
            if p == 0:
                # Writing null spectrum data
                fv.write("Null: \n")
                fm.write("Null: \n")
                fs.write("Null: \n")
                ft.write("Null: \n")

            else:
                # Writing spectrum data
                vstr = v.tolist()
                values = str(vstr) + ":\n"
                fv.write(values)

                # Writing mass data
                mstr = m.tolist()
                values = str(mstr) + ":\n"
                fm.write(values)

                # Writing SFR data
                sstr = s.tolist()
                values = str(sstr) + ":\n"
                fs.write(values)

                # Writing temp data
                tstr = t.tolist()
                values = str(tstr) + ":\n"
                ft.write(values)

    else:
        # Priting a null array if no particles
        rowList = [gid, mass, sfr, "False", "Null", "Null", "Null", "Null", "Null"]
        writer.writerow(rowList)

        # Writing spectrum data
        fv.write("Null: \n")
        fm.write("Null: \n")
        fs.write("Null: \n")
        ft.write("Null: \n")

    # Updating i
    i = i + 1

# Closing the files
fv.close()
fm.close()
fs.close()
ft.close()
f.close()

print("Done")