# Importing libraries
import numpy as np
import math

# Function to rotate the galaxy about a point
def rotateAboutVector(angle, rotationVector, x, y, z):

    c = math.cos(angle)
    s = math.sin(angle)

    ux = rotationVector[0]
    uy = rotationVector[1]
    uz = rotationVector[2]

    newx = (c + ux*ux * (1-c)) * x + (ux * uy * (1-c) - uz * s) * y + (ux * uz * (1-c) + uy * s) * z
    newy = (uy * ux * (1-c) + uz * s) * x + (c + uy*uy * (1-c)) * y + (uy * uz * (1-c) - ux * s) * z
    newz = (uz * ux * (1-c) - uy * s) * x + (uz * uy * (1-c) + ux * s) * y + (c + uz*uz * (1-c)) * z

    return newx, newy, newz

# Function to get the vector from two spherical coordinate angles
def getVectorFromAngle(theta, phi):

    # Calculating each vector component
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Returning the vector
    return np.array([x, y, z])

# Function to get the cylinder vectors
def getCylinders(nCyl):
    # Defining the theta and phi angles to get this many cylinders
    theta = np.linspace(0, np.pi, int(np.sqrt(nCyl)), endpoint=True)
    phi = np.linspace(0, 2*np.pi, int(np.sqrt(nCyl)), endpoint=True)

    # Generating the cylinder vectors
    vectors = []
    inclinations = []
    angles = []
    thetas = []

    for i in range(len(theta)):
        if theta[i] == 0 or theta[i] == np.pi:
            vectors.append(getVectorFromAngle(theta[i], 0))
            inclinations.append(0.0)
            angles.append(0.0)
            thetas.append(np.round(theta[i]*(360/(2*np.pi))))
        else:
            for j in range(len(phi)):
                vectors.append(getVectorFromAngle(theta[i], phi[j]))

                if theta[i] < (np.pi/2):
                    inclinations.append(np.round(theta[i]*(360/(2*np.pi))))
                elif theta[i] > (np.pi/2):
                    inclinations.append(np.round(90 - (theta[i]*(360/(2*np.pi))-90)))
                else:
                    inclinations.append(int(90))

                angles.append(np.round(phi[j]*(360/(2*np.pi))))
                thetas.append(np.round(theta[i]*(360/(2*np.pi))))

    return vectors, inclinations, angles, thetas