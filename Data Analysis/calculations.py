# Importing libraries
import numpy as np
import pandas as pd
import astropy.units as u

# Function to calculate the star formation rate density
def sfrDensity(starFormationRate, radius):
    return (starFormationRate) / (np.pi * (radius)**2)

# Function to calculate stellar density
def sDensity(stellarMass, radius):
    return (stellarMass) / (np.pi * (radius)**2)

# Function to calculate rotation velocity  (S. McGaugh 2011)
def rotationVelocity(stellarMass):
    A = 47 * u.Msun * u.km**-4 * u.s**4
    return (stellarMass / A)**(1/4)

# Function to calculate mass loading factor
def massLoadingFactor(starFormationRate, massOutflowRate):
    return massOutflowRate / starFormationRate