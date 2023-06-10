# Imports
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import skew
import astropy.units as u

def readSpectrum(velsFilename, massFilename):
    # Opening files, loading in data and closing them
    vf = open(velsFilename, "r")
    vels = vf.readlines()
    vf.close()

    mf = open(massFilename, "r")
    mass = mf.readlines()
    mf.close()

    # Turning the data into an array
    specData = pd.DataFrame(vels)
    specData = specData[0]

    massData = pd.DataFrame(mass)
    massData = massData[0]

    return specData, massData

def readParticleFiles(filename):
    f = open(filename, "r")
    data = f.readlines()
    f.close()

    data = pd.DataFrame(data)
    data = data[0]

    return data

def binVelocities(velsFilename, massFilename, dataFrame, bins):

    # Getting the velocity data
    specData, massData = readSpectrum(velsFilename, massFilename)

    # Looping through every line in the data and histogramming it
    for i in tqdm(range(len(specData))):
        # Stripping fomratting
        data = specData[i].strip("][").split(",")
        mdat = massData[i].strip("][").split(",")

        # Putting zeros for bins if no data
        if data == ["Null: \n"]:
            histogramData = np.zeros(len(bins)-1)
        else:
            # Stripping end formatting
            data = data[:-2]
            mdat = mdat[:-2]

            # Weighting the data
            data = np.array(data, dtype=float)
            mdat = np.array(mdat, dtype=float)

            histogramData = np.histogram(data, bins=bins, weights=mdat)
            histogramData = histogramData[0]

        specData[i] = histogramData

    # Setting the dataframe column
    dataFrame["Spectra"] = specData

    return dataFrame

def removeNulls(df):
    # Only getting galaxies with particles
    df = df[df["Particles"] == True]
    df["N Particles"] = pd.to_numeric(df["N Particles"])
    df = df[df["N Particles"] > 0]

    return df

def binGalaxy(df, binNum):
    # Only getting galaxies with particles
    df = removeNulls(df)

    df = df[df["Star Formation Rate"] > 0]

    # Logging the mass and star formation rates
    df["Mass"] = np.log10(df["Mass"])
    df["Star Formation Rate"] = np.log10(df["Star Formation Rate"])

    # Getting the bins
    h, xedges, yedges = np.histogram2d(df["Mass"], df["Star Formation Rate"], bins=binNum)
    df["Binned Mass"], massBinVals = pd.cut(df["Mass"], xedges, labels=False, retbins=True)
    df["Binned Sfr"], sfrBinVals = pd.cut(df["Star Formation Rate"], yedges, labels=False, retbins=True)

    # Getting the usable bin values
    massBins = np.unique(df["Binned Mass"])
    sfrBins = np.unique(df["Binned Sfr"])
    massBins = massBins[:-1]
    sfrBins = sfrBins[:-1]

    return df, massBins, sfrBins, massBinVals, sfrBinVals

def createBinnedFiles(velsFilename, massFilename, dataFrame, binNum):

    # Getting the velocity data
    specData, massData = readSpectrum(velsFilename, massFilename)

    # Binning the masses and star formation rates
    _, massBins, sfrBins, massBinVals, sfrBinVals = binGalaxy(dataFrame, binNum)

    # Getting ranges
    massRange = massBinVals[1] - massBinVals[0]
    sfrRange = sfrBinVals[1] - sfrBinVals[0]

    massBinValsMax = massBinVals + massRange
    sfrBinValsMax = sfrBinVals + sfrRange

    # Looping through each mass and sfr bin
    for s in range(binNum):
        for m in range(binNum):

            # Creating the filename
            filenameV = "binnedData/M{:.2f}".format(massBinVals[m]) + "S{:.2f}".format(sfrBinVals[s]) + "V.txt"
            filenameM = "binnedData/M{:.2f}".format(massBinVals[m]) + "S{:.2f}".format(sfrBinVals[s]) + "M.txt"

            # Creating arrays we want to write 
            particleVels = np.array([])
            particleMass = np.array([])

            # Running through velocity data
            for i in tqdm(range(len(specData))):

                # Checking if this i is in our bins
                d = dataFrame.iloc[i]
                mass = np.log10(d["Mass"])
                sfr = np.log10(d["Star Formation Rate"])

                if mass >= massBinVals[m] and mass <= massBinValsMax[m]:
                    if sfr >= sfrBinVals[s] and sfr <= sfrBinValsMax[s]:
                        # Stripping formatting
                        data = specData[i].strip("][").split(",")
                        mdat = massData[i].strip("][").split(",")

                        # Passing if no data
                        if data == ["Null: \n"]:
                            pass
                        else:
                            # Stripping end formatting
                            data = data[:-2]
                            mdat = mdat[:-2]

                            # Turning into arrays
                            data = np.array(data, dtype=float)
                            mdat = np.array(mdat, dtype=float)

                            # Adding onto our array
                            particleVels = np.append(particleVels, data)
                            particleMass = np.append(particleMass, mdat)

            # Writing the arrays to our file
            np.savetxt(filenameV, particleVels)
            np.savetxt(filenameM, particleMass)
    
def getBinStats(massBinVals, sfrBinVals, binNum):

    # Creating arrays to store the values
    means = np.zeros((binNum, binNum))
    stds = np.zeros_like(means)
    skews = np.zeros_like(means)

    # Reversing the sfr bins to get right plot ordering
    ss = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    
    # Looping through all m and s
    for s in tqdm(range(binNum)):
        for m in range(binNum):

            # Opening the corresponding file
            filenameV = "binnedData/M{:.2f}".format(massBinVals[m]) + "S{:.2f}".format(sfrBinVals[s]) + "V.txt"
            filenameM = "binnedData/M{:.2f}".format(massBinVals[m]) + "S{:.2f}".format(sfrBinVals[s]) + "M.txt"

            velsData = np.loadtxt(filenameV)
            massData = np.loadtxt(filenameM)

            # Calculating the stats
            if len(velsData) == 0:
                means[ss[s]][m] = float("nan")
                stds[ss[s]][m] = float("nan")
                skews[ss[s]][m] = float("nan")
            else:
                means[ss[s]][m] = np.average(velsData, weights=massData)
                stds[ss[s]][m] = np.std(velsData)
                skews[ss[s]][m] = skew(velsData)

    return means, stds, skews

def massOutflowRates(velsFilename, massFilename, galaxyData):

    # Creating empty array for the mass outflow rate
    massOutflow = []
    outflowVel = []

    # Reading in the veloicty and mass spectrum data
    velocities, masses = readSpectrum(velsFilename, massFilename)

    # Looping through all the galaxy detections
    for i in tqdm(range(len(velocities))):
        
        # Getting the current galaxies rotation velocity
        d = galaxyData.iloc[i]
        vRot = d["Rotation Velocity"]

        # Calculating the range we want to check outside of
        vRange = 2 * vRot

        # Stripping and checking the velocity data
        vels = velocities[i].strip("][").split(",")
        mdat = masses[i].strip("][").split(",")

        # Putting zeros for bins if no data
        if vels == ["Null: \n"]:
            massOutflow.append(0)
            outflowVel.append(0)
        else:
            # Stripping end formatting
            vels = vels[:-2]
            mdat = mdat[:-2]

            vels = np.array(vels, dtype=float)
            mdat = np.array(mdat, dtype=float)

            # Getting the values outside our given range
            absoluteVels = np.abs(vels)
            velsFlowing = vels[absoluteVels>=vRange]

            if len(velsFlowing) == 0:
                massOutflow.append(0)
                outflowVel.append(0)
            else:
                # Selecting the masses outflowing
                mass = mdat[absoluteVels>=vRange]

                # Calculating the outflow rate
                #outflow = np.sum(velsFlowing * mass) / (3 * 3.09e13)
                #massOutflow.append(outflow)
                outflowVel.append(np.mean(velsFlowing))

                # Calculating the outflow rate
                averageVelocity = np.mean(velsFlowing)
                totalMass = np.sum(mass)
                outflow = averageVelocity * totalMass / (3 * 1e3 * 3.09e13)
                massOutflow.append(outflow)

    # Returning the outflow rates
    galaxyData["Mass Outflow Rate"] = massOutflow
    galaxyData["Outflow Velocity"] = outflowVel

    return galaxyData

def oneCylinderDetection(theta, phi):
    # Reading in the galaxy
    galaxyData = pd.read_csv("galaxyDataSave2.csv", delimiter=",")

    # Loading the particle data
    v, m = readSpectrum("linuxData/velocities.txt", "linuxData/masses.txt")

    # Adding particle data to the dataframe
    galaxyData["Velocities"] = v
    galaxyData["Masses"] = m

    # Removing nulls 
    galaxyData = removeNulls(galaxyData)

    # Choosing only observations with one theta and phi value
    galaxyDataCut = galaxyData[galaxyData["Theta"] == str(theta)]
    galaxyDataCut = galaxyDataCut[galaxyDataCut["Phi"] == str(phi)]

    # Binning the galaxy
    galaxyData, massBins, sfrBins, massBinVals, sfrBinVals = binGalaxy(galaxyDataCut, 10)

    # Looping through all the bins to get the means
    means = np.zeros((10,10))
    nums = np.zeros_like(means)

    # Reversing the sfr bins to get right plot ordering
    ss = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    # Looping through all m and s
    for s in tqdm(range(10)):
        for m in range(10):

            # Slicing our dataframe
            data = galaxyData[galaxyData["Binned Sfr"] == s]
            data = data[data["Binned Mass"] == m]

            # Creating empty arrays
            vels = np.array([])
            mass = np.array([])

            # Extracting the velocities and masses
            for i in range(len(data)):
                # Getting the row
                d = data.iloc[i]

                # Extracting velocity and mass data
                vData = d["Velocities"]
                mData = d["Masses"]

                # Stripping formatting
                vData = vData[:-2]
                mData = mData[:-2]
                vData = vData.strip("][").split(",")
                mData = mData.strip("][").split(",")

                # Putting into array form
                vData = np.array(vData, dtype=float)
                mData = np.array(mData, dtype=float)

                # Appending to the end of the vels and mass arrays
                vels = np.append(vels, vData)
                mass = np.append(mass, mData)

            # Calculating the mean
            if len(vels) == 0:
                means[ss[s]][m] = float("nan")
                nums[ss[s]][m] = float("nan")
            else:
                means[ss[s]][m] = np.average(vels, weights=mass)
                nums[ss[s]][m] = len(data)

    return galaxyData, means, nums, massBins, sfrBins, massBinVals, sfrBinVals

def loadEverything(selection, theta, phi, filename="galaxyDataSave2.csv", vfile="linuxData/velocities.txt", mfile="linuxData/masses.txt", sfile="linuxData/sfrs.txt", tfile="linuxData/temperatures.txt"):
    # Loading in the galaxy data
    galaxyData = pd.read_csv(filename, delimiter=",")

    # Loading the velocity, mass, temperature and sfr data in too
    velocityData = readParticleFiles(vfile)
    massData = readParticleFiles(mfile)
    sfrData = readParticleFiles(sfile)
    tempData = readParticleFiles(tfile)

    # Appending these to our dataframe
    galaxyData["ParticleV"] = velocityData
    galaxyData["ParticleM"] = massData
    galaxyData["ParticleS"] = sfrData
    galaxyData["ParticleT"] = tempData

    # If making a sample selection, do it here
    if selection == "ANGLE":
        galaxyDataCut = galaxyData[galaxyData["Inclination"] == str(theta)]
        #galaxyData = galaxyDataCut[galaxyDataCut["Phi"] == str(phi)]
        galaxyData = galaxyDataCut
    else:
        pass

    return galaxyData

def binEverything(selection, theta, phi, nBins, thresholdType, threshhold, filename="galaxyDataSave2.csv", vfile="linuxData/velocities.txt", mfile="linuxData/masses.txt", sfile="linuxData/sfrs.txt", tfile="linuxData/temperatures.txt"):

    # Loading in the galaxy data and particle data 
    galaxyData = loadEverything(selection, theta, phi, filename, vfile, mfile, sfile, tfile)

    # Binning the galaxy into mass and sfr bins
    galaxyData, massBins, sfrBins, massBinVals, sfrBinVals = binGalaxy(galaxyData, nBins)

    # List to get sfr ordering correct
    #ss = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    # Creating empty arrays for any summary statistics we want
    means = np.zeros((nBins, nBins))
    nGals = np.zeros_like(means)

    # Looping through all the bins
    for s in tqdm(range(nBins)):
        for m in range(nBins):

            # Slice the dataframe by the s and m value
            data = galaxyData[galaxyData["Binned Sfr"] == s]
            data = data[data["Binned Mass"] == m]

            # Creating arrays for all the particle properties
            vels = np.array([])
            mass = np.array([])
            sfrs = np.array([])
            temp = np.array([])

            # Looping through all the individual detections in this bin
            for i in range(len(data)):
                # Selecting the ith galaxy
                d = data.iloc[i]

                # Extracting the particle properties
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

                # Appending to the final arrays
                vels = np.append(vels, gv)
                mass = np.append(mass, gm)
                sfrs = np.append(sfrs, gs)
                temp = np.append(temp, gt)

            if len(data) == 0:
                #means[ss[s]][m] = float("nan")
                #nGals[ss[s]][m] = float("nan")
                means[s][m] = float("nan")
                nGals[s][m] = float("nan")
            else:
                # Performing any selections
                if thresholdType == "SFR":
                    vels = vels[sfrs > threshhold]
                    mass = mass[sfrs > threshhold]
                elif thresholdType == "TEMP":
                    vels = vels[temp < threshhold]
                    mass = mass[temp < threshhold]
                else:
                    pass

                # Checking if there are still particles 
                if len(vels) == 0:
                    #means[ss[s]][m] = float("nan")
                    #nGals[ss[s]][m] = float("nan")
                    means[s][m] = float("nan")
                    nGals[s][m] = float("nan")
                else:

                    # Calculating the weighted mean
                    #means[ss[s]][m] = np.average(vels, weights=mass)
                    #nGals[ss[s]][m] = len(data)
                    means[s][m] = np.average(vels, weights=mass)
                    nGals[s][m] = len(data)
                
    # Returning everything we need
    return galaxyData, means, nGals, massBins, massBinVals, sfrBins, sfrBinVals

def particleFractions(selection, theta, phi, nBins, nV):

    # Loading data
    galaxyData = loadEverything(selection, theta, phi)

    # Binning the galaxy into mass and sfr bins
    galaxyData, massBins, sfrBins, massBinVals, sfrBinVals = binGalaxy(galaxyData, nBins)

    # List to get sfr ordering correct
    ss = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

    # Creating lists to store info we want
    outflowFracN = np.zeros((nBins, nBins))
    outflowFracM = np.zeros_like(outflowFracN)
    inflowFracN = np.zeros_like(outflowFracN)
    inflowFracM = np.zeros_like(outflowFracN)
    nonDetectionN = np.zeros_like(outflowFracN)
    nonDetectionM = np.zeros_like(outflowFracN)

    # Looping through all the bins
    for s in tqdm(range(nBins)):
        for m in range(nBins):

            # Slice the dataframe by the s and m value
            data = galaxyData[galaxyData["Binned Sfr"] == s]
            data = data[data["Binned Mass"] == m]

            # Empty data to store 
            vIn = np.array([])
            vOut = np.array([])
            vNon = np.array([])
            mIn = np.array([])
            mOut = np.array([])
            mNon = np.array([])
            nTot = 0
            mTot = 0

            # Looping through each galaxy in the bin
            for i in range(len(data)):
                # Selecting the ith galaxy
                d = data.iloc[i]

                # Extracting the particle properties
                gv = d["ParticleV"]
                gm = d["ParticleM"]

                # Stripping all the formatting
                gv = gv[:-2]
                gm = gm[:-2]

                gv = gv.strip("][").split(",")
                gm = gm.strip("][").split(",")

                # Putting into arrays
                gv = np.array(gv, dtype=float)
                gm = np.array(gm, dtype=float)

                # Getting the threshold velocity
                vThresh = d["Rotation Velocity"]

                # Appending particles that are outflowing
                vOut = np.append(vOut, gv[gv > nV*vThresh])
                mOut = np.append(mOut, gm[gv > nV*vThresh])

                # Appending particles that are inflowing
                vIn = np.append(vIn, gv[gv < -nV*vThresh])
                mIn = np.append(mIn, gm[gv < -nV*vThresh])

                # Appending particles that are not doing either
                v1 = gv[gv < nV*vThresh]
                m1 = gm[gv < nV*vThresh]
                v2 = v1[v1 > -nV*vThresh]
                m2 = m1[v1 > -nV*vThresh]
                vNon = np.append(vNon, v2)
                mNon = np.append(mNon, m2)

                # Adding to the total number of particles and total mass
                nTot = nTot + len(gv)
                mTot = mTot + np.sum(gm)

            # Getting our final stats
            if nTot == 0:
                outflowFracN[ss[s]][m] = float("nan")
                outflowFracM[ss[s]][m] = float("nan")
                inflowFracN[ss[s]][m] = float("nan")
                inflowFracM[ss[s]][m] = float("nan")
                nonDetectionN[ss[s]][m] = float("nan")
                nonDetectionM[ss[s]][m] = float("nan")
            else:
                if len(vOut) == 0: 
                    outflowFracN[ss[s]][m] = 0
                    outflowFracM[ss[s]][m] = 0
                else:
                    outflowFracN[ss[s]][m] = len(vOut) / nTot
                    outflowFracM[ss[s]][m] = np.sum(mOut) / mTot

                if len(vIn) == 0:
                    inflowFracN[ss[s]][m] = 0
                    inflowFracM[ss[s]][m] = 0
                else:
                    inflowFracN[ss[s]][m] = len(vIn) / nTot
                    inflowFracM[ss[s]][m] = np.sum(mIn) / mTot

                if len(vNon) == 0:
                    nonDetectionN[ss[s]][m] = 0
                    nonDetectionM[ss[s]][m] = 0
                else:
                    nonDetectionN[ss[s]][m] = len(vNon) / nTot
                    nonDetectionM[ss[s]][m] = np.sum(mNon) / mTot

    return galaxyData, outflowFracN, outflowFracM, inflowFracN, inflowFracM, nonDetectionN, nonDetectionM, massBins, sfrBins, massBinVals, sfrBinVals










