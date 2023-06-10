# Importing libraries
import numpy as np
from dataAnalysisFunctions import *
from tqdm import tqdm
import uncertainties as uc
from uncertainties import unumpy

# Defining the function to get outflow properties for each galaxy
def galaxyOutflowProperties(selection, theta, phi, nV, file="galaxyDataSave1.csv", vfile="linuxData/velocities.txt", mfile="linuxData/masses.txt", sfile="linuxData/sfrs.txt", tfile="linuxData/temperatures.txt"):

    # Loading in the galaxy and particle data 
    galaxyData = loadEverything(selection, theta, phi, file, vfile, mfile, sfile, tfile)

    # Removing null detections
    galaxyData = removeNulls(galaxyData)

    # Defining arrays for each of the properties we want
    outflowVelocity = []
    outflowMass = []
    outflowTemp = []
    outflowSFR = []

    inflowVelocity = []
    inflowMass = []
    inflowTemp = []
    inflowSFR = []

    # Defining error arrays
    outflowVelErrors = []
    outflowMassErrors = []
    outflowTempErrors = []

    inflowVelErrors = []
    inflowMassErrors = []
    inflowTempErrors = []

    # Looping through every galaxy detection
    for i in tqdm(range(len(galaxyData))):

        # Getting the current galaxy
        d = galaxyData.iloc[i]

        # Getting the rotation velocity of this galaxy and threshold velocity
        vRot = d["Rotation Velocity"]
        vThresh = nV * vRot

        # Getting the velocities, masses and temperatures 
        gv = d["ParticleV"]
        gm = d["ParticleM"]
        gs = d["ParticleS"]
        gt = d["ParticleT"]

        # Stripping all the formatting
        gv = gv[:-2]
        gm = gm[:-2]
        gs = gs[:-2]
        gt = gt[:-2]

        gv = gv.strip("][").split(",")
        gm = gm.strip("][").split(",")
        gs = gs.strip("][").split(",")
        gt = gt.strip("][").split(",")

        # Putting into arrays
        gv = np.array(gv, dtype=float)
        gm = np.array(gm, dtype=float)
        gs = np.array(gs, dtype=float)
        gt = np.array(gt, dtype=float)

        # Selecting outflows
        if len(gv[gv > vThresh]) != 0:
            averageVelocity = np.average(gv[gv > vThresh], weights = gm[gv > vThresh])
            totalMass = np.sum(gm[gv > vThresh])
            outflowRate = totalMass * averageVelocity / (3 * 1e3 * 3.09e13)

            outflowVelocity.append(averageVelocity)
            outflowMass.append(outflowRate * (365 * 24 * 60 * 60))
            outflowTemp.append(np.median(gt[gv > vThresh]))
            outflowSFR.append(np.sum(gs[gv > vThresh]))

            velError = np.std(gv[gv > vThresh])
            outflowVelErrors.append(velError)
            outflowMassErrors.append(365*24*60*60*velError * totalMass / (3 * 1e3 * 3.09e13))
            outflowTempErrors.append(np.std(gt[gv > vThresh]))
        else:
            outflowVelocity.append(0)
            outflowMass.append(0)
            outflowTemp.append(0)
            outflowSFR.append(0)

            outflowVelErrors.append(0)
            outflowMassErrors.append(0)
            outflowTempErrors.append(0)

        # Selecting inflows
        if len(gv[gv < -1*vThresh]) != 0:
            averageVelocity = np.average(gv[gv < -1*vThresh], weights = gm[gv < -1*vThresh])
            totalMass = np.sum(gm[gv < -1*vThresh])
            inflowRate = totalMass * averageVelocity / (3 * 1e3 * 3.09e13)

            inflowVelocity.append(averageVelocity)
            inflowMass.append(inflowRate * (365 * 24 * 60 * 60)) 
            inflowTemp.append(np.median(gt[gv < -1*vThresh]))
            inflowSFR.append(np.sum(gs[gv < -1*vThresh]))

            velError = np.std(gv[gv < -1*vThresh])
            inflowVelErrors.append(velError)
            inflowMassErrors.append(365*24*60*60*velError * totalMass / (3 * 1e3 * 3.09e13))
            inflowTempErrors.append(np.std(gt[gv < -1*vThresh]))
        else:
            inflowVelocity.append(0)
            inflowMass.append(0)
            inflowTemp.append(0)
            inflowSFR.append(0)

            inflowVelErrors.append(0)
            inflowMassErrors.append(0)
            inflowTempErrors.append(0)

    # Dropping the columns we don't need
    galaxyData = galaxyData.drop("ParticleV", axis="columns")
    galaxyData = galaxyData.drop("ParticleM", axis="columns")
    galaxyData = galaxyData.drop("ParticleT", axis="columns")
    galaxyData = galaxyData.drop("ParticleS", axis="columns")

    # Assigning values to the galaxy data array
    galaxyData["Outflow Velocity"] = outflowVelocity
    galaxyData["Outflow Velocity Error"] = outflowVelErrors

    galaxyData["Mass Outflow Rate"] = outflowMass 
    galaxyData["Mass Outflow Rate Error"] = outflowMassErrors

    galaxyData["Outflow Temperature"] = outflowTemp
    galaxyData["Outflow Temperature Error"] = outflowTempErrors

    galaxyData["Outflow SFR"] = outflowSFR

    galaxyData["Inflow Velocity"] = inflowVelocity
    galaxyData["Inflow Velocity Error"] = inflowVelErrors

    galaxyData["Mass Inflow Rate"] = inflowMass 
    galaxyData["Mass Inflow Rate Error"] = inflowMassErrors

    galaxyData["Inflow Temperature"] = inflowTemp
    galaxyData["Inflow Temperature Error"] = inflowTempErrors

    galaxyData["Inflow SFR"] = inflowSFR

    # Returning the arrays
    return galaxyData

# Defining function to bin the outflow and inflow data for plotting
def binFlowData(galaxyData, nBins=10, xaxis="MASS"):

    # Checking if we're plotting against mass or sfr
    if xaxis == "MASS":
        # Extracting our masses
        galaxyData["Mass"] = np.log10(galaxyData["Mass"])
        galaxyMasses = galaxyData["Mass"].to_numpy()

        # Binning the data based on mass
        bins = np.linspace(min(galaxyMasses), max(galaxyMasses), nBins)
        galaxyData["Bins"], binVals = pd.cut(galaxyData["Mass"], bins, labels=False, retbins=True)

    elif xaxis == "SFR":
        # Selecting only galaxies with SFR > 0 (avoid log errors)
        galaxyData = galaxyData[galaxyData["Star Formation Rate"] > 0]

        # Extracting star formation rates
        galaxyData["Star Formation Rate"] = np.log10(galaxyData["Star Formation Rate"])
        galaxySfrs = galaxyData["Star Formation Rate"].to_numpy()

        # Binning the data based on sfrs
        bins = np.linspace(min(galaxySfrs), max(galaxySfrs), nBins)
        galaxyData["Bins"], binVals = pd.cut(galaxyData["Star Formation Rate"], bins, labels=False, retbins=True)

    # Getting the bin indicies
    bins = galaxyData["Bins"]
    bins = np.unique(bins[~np.isnan(bins)])

    # Creating empty arrays to store averages
    oV = []; oM = []; oT = []
    iV = []; iM = []; iT = []

    # Finding the average value in each bin for all bins
    for i in range(len(bins)):

        # Selecting galaxies in this bin
        sample = galaxyData[galaxyData["Bins"] == bins[i]]

        if len(sample) == 0:
            oV.append(np.float("nan")); oM.append(np.float("nan")); oT.append(np.float("nan"))
            iV.append(np.float("nan")); iM.append(np.float("nan")); iT.append(np.float("nan"))
        else:
            # Creating uncertaintity objects
            v = unumpy.uarray(sample["Outflow Velocity"].to_numpy(), sample["Outflow Velocity Error"].to_numpy())
            m = unumpy.uarray(sample["Mass Outflow Rate"].to_numpy(), sample["Mass Outflow Rate Error"].to_numpy())
            t = unumpy.uarray(sample["Outflow Temperature"].to_numpy(), sample["Outflow Temperature Error"].to_numpy())

            # Only choosing nonzero values
            v = v[v != 0]
            m = m[m != 0]
            t = t[t != 0]

            oV.append(np.mean(v))
            oM.append(np.mean(m))
            oT.append(np.mean(t))

            v = unumpy.uarray(sample["Inflow Velocity"].to_numpy(), sample["Inflow Velocity Error"].to_numpy())
            m = unumpy.uarray(sample["Mass Inflow Rate"].to_numpy(), sample["Mass Inflow Rate Error"].to_numpy())
            t = unumpy.uarray(sample["Inflow Temperature"].to_numpy(), sample["Inflow Temperature Error"].to_numpy())

            # Only choosing nonzero values
            v = v[v != 0]
            m = m[m != 0]
            t = t[t != 0]

            iV.append(np.mean(v))
            iM.append(np.mean(m))
            iT.append(np.mean(t))

    # Returning
    return binVals, oV, iV, oM, iM, oT, iT