# Importing libraries
import numpy as np
import h5py
from pyread_eagle import EagleSnapshot
from geometricFunctions import rotateAboutVector

# Attribute values
gasAttrs = ["GroupNumber", "SubGroupNumber", "Coordinates", "Velocity", "Mass", "StarFormationRate"]
starAttrs = ["GroupNumber", "SubGroupNumber", "Coordinates", "Velocity", "Mass"]

Nreject = 50
Sreject = 10

# Defining the galaxy class
class galaxy:

    # Initialisation function
    def __init__(self, gn, sgn, centre, velocity, size, stars, filepath):

        # Getting the filepath variable
        self.filepath = filepath

        # Getting the header parameters
        self.a, self.h, self.boxsize = self.read_header()

        # Loading in the gas data we need (only bound particles)
        self.gasDataBound = self.read_galaxy(0, gn, sgn, centre, size, True, gasAttrs)

        # Checking if particles are present
        if len(self.gasDataBound["Mass"]) < Nreject:
            self.particles = False
        else: 
            self.particles = True

            # Loading in the unbound gas data too
            self.gasDataBox = self.read_galaxy(0, gn, sgn, centre, size, False, gasAttrs)

            # Decomposing the properties into separate components
            self.gasArrayLength = len(self.gasDataBox["Mass"])
            self.gx  = self.gasDataBox["Coordinates"][0:self.gasArrayLength, 0] 
            self.gy  = self.gasDataBox["Coordinates"][0:self.gasArrayLength, 1]
            self.gz  = self.gasDataBox["Coordinates"][0:self.gasArrayLength, 2]
            self.gvx = self.gasDataBox["Velocity"][0:self.gasArrayLength, 0]
            self.gvy = self.gasDataBox["Velocity"][0:self.gasArrayLength, 1]
            self.gvz = self.gasDataBox["Velocity"][0:self.gasArrayLength, 2]
            self.gm  = self.gasDataBox["Mass"]
            self.sfr = self.gasDataBox["StarFormationRate"]

            # Loading in the star particle data if wanted
            if stars:
                self.starDataBox = self.read_galaxy(4, gn, sgn, centre, size, False, starAttrs)
                self.starArrayLength = len(self.starDataBox["Mass"])
                self.sx  = self.starDataBox["Coordinates"][0:self.starArrayLength, 0] 
                self.sy  = self.starDataBox["Coordinates"][0:self.starArrayLength, 1]
                self.sz  = self.starDataBox["Coordinates"][0:self.starArrayLength, 2]
                self.svx = self.starDataBox["Velocity"][0:self.starArrayLength, 0]
                self.svy = self.starDataBox["Velocity"][0:self.starArrayLength, 1]
                self.svz = self.starDataBox["Velocity"][0:self.starArrayLength, 2]
                self.sm  = self.starDataBox["Mass"]

            # Defining the peculiar veloicty of the galaxy
            self.veloicty = velocity

            # Orienting the galaxy
            self.orient_galaxy(stars)

            # Calculating the radius of all particles from the centre 
            self.scaleFactor = np.sqrt(self.gx**2 + self.gy**2 + self.gz**2)

            # Dropping bound gas data from the memory
            del self.gasDataBound

    # Function to read the Eagle file headers
    def read_header(self):
        # Read in various attributes from the header group
        #f = h5py.File('./data/snap_028_z000p000.0.hdf5', 'r')
        f = h5py.File(self.filepath, "r")

        # The Scale Factor
        a = f['Header']. attrs.get('Time')

        # The Hubble Parameter
        h = f['Header']. attrs.get('HubbleParam')

        # The Simulation Box Size
        boxsize = f['Header']. attrs.get('BoxSize') 

        # Closing the file
        f.close()

        # Returning values
        return a, h, boxsize

    # Function to read in data from the Eagle files
    def read_galaxy(self, itype, gn, sgn, centre, size, maskBool, attrs):

        # Setting things up
        data = {}
        centrePoint = centre * self.h

        # Initialising the read eagle routine
        #eagle_data = EagleSnapshot("./data/snap_028_z000p000.0.hdf5")
        eagle_data = EagleSnapshot(self.filepath)

        # Defining the region to load
        region = np.array([
            (centrePoint[0] - 0.5 * size), (centrePoint[0] + 0.5 * size),
            (centrePoint[1] - 0.5 * size), (centrePoint[1] + 0.5 * size),
            (centrePoint[2] - 0.5 * size), (centrePoint[2] + 0.5 * size)])

        # Telling eagle what region we want and loading it in
        eagle_data.select_region(*region)
        #f = h5py.File("./data/snap_028_z000p000.0.hdf5", "r")
        f = h5py.File(self.filepath, "r")

        # Looping through all the things we want to load in
        for att in attrs:
            tmp = eagle_data.read_dataset(itype, att)
            cgs =  f["PartType%i/%s" % (itype, att)].attrs.get("CGSConversionFactor")
            aexp = f["PartType%i/%s" % (itype, att)].attrs.get("aexp-scale-exponent")
            hexp = f["PartType%i/%s" % (itype, att)].attrs.get("h-scale-exponent")

            # Periodic wrap coordinates around centre
            if att == "Coordinates": 
                tmp = np.mod(tmp - centrePoint + 0.5 * self.boxsize, self.boxsize) + centrePoint - 0.5 * self.boxsize
                
                # Centering the points around the origin
                tmp = tmp - centrePoint

            # Converting to proper units   
            data[att] = np.multiply(tmp, cgs * self.a**aexp * self.h**hexp, dtype=np.longfloat)
        f.close()

        # Getting only bound particles based on mask boolean
        if maskBool:
            mask = np.logical_and(data["GroupNumber"] == gn, data["SubGroupNumber"] == sgn)
            for att in data.keys():
                data[att] = data[att][mask]

        # Returning the data 
        return data

    # Function to orient the galaxy according to its angular momentum vector
    def orient_galaxy(self, stars):

        # Getting the star formation rate
        sfr = self.gasDataBound["StarFormationRate"]

        if len(sfr[sfr>0]) <= Sreject:
            self.particles = False
        else:
            # Getting the masses 
            masses = (self.gasDataBound["Mass"]).reshape(len(self.gasDataBound["Mass"]), 1)
            masses = masses[sfr > 0]

            # Getting the positions and velocities
            positions = self.gasDataBound["Coordinates"][sfr > 0]
            velocities = self.gasDataBound["Velocity"][sfr > 0]

            # Calculating the angular momentum
            momentum = np.cross(positions, velocities)
            angular = momentum * masses
            total = np.sum(angular, axis=0)

            # Normalising the vector
            self.angularMomentum = total / np.linalg.norm(total)

            # Getting the angle to rotate by
            angle = np.arccos(np.dot(self.angularMomentum, np.array([0,0,1])))

            # Getting a perpendicular vector to rotate around and normalising it
            rotationVector = np.cross(self.angularMomentum, np.array([0,0,1]))
            rotationVector = rotationVector / np.linalg.norm(rotationVector)

            # Rotating the position and velocity vectors of the gas
            self.gx, self.gy, self.gz = rotateAboutVector(angle, rotationVector, self.gx, self.gy, self.gz)
            self.gvx, self.gvy, self.gvz = rotateAboutVector(angle, rotationVector, self.gvx, self.gvy, self.gvz)

            # Rotating the galaxies velocity
            self.veloicty[0], self.veloicty[1], self.veloicty[2] = rotateAboutVector(angle, rotationVector, self.veloicty[0], self.veloicty[1], self.veloicty[2])

            # Rotating the stars as well
            if stars:
                self.sx, self.sy, self.sz = rotateAboutVector(angle, rotationVector, self.sx, self.sy, self.sz)
                self.svx, self.svy, self.svz = rotateAboutVector(angle, rotationVector, self.svx, self.svy, self.svz)

    # Function to get a line of sight cylinder
    def cylinder(self, vector, size):

        # Normalising the vector
        vectorNorm = vector / np.linalg.norm(vector)

        # Finding the radial distance of each point from the cylinder vector
        sphereEquation = np.sqrt((self.gx - vectorNorm[0]*self.gx)**2 + (self.gy - vectorNorm[1]*self.gy)**2 + (self.gz - vectorNorm[2]*self.gz)**2)

        # Only choosing points that lie within a sphere centered around a point on the cylinder line
        inCylinder = sphereEquation <= size

        # Only choosing those above the axis or below the axis
        if vector[2] > 0:
            height = self.gz >= 0
        elif vector[2] < 0:
            height = self.gz <= 0
        else:
            if vector[0] > 0 and vector[1] > 0:
                height = (self.gx >= 0) * (self.gy >= 0)
            elif vector[0] > 0 and vector[1] < 0:
                height = (self.gx >= 0) * (self.gy <= 0)
            elif vector[0] < 0 and vector[1] > 0:
                height = (self.gx <= 0) * (self.gy <= 0)
            elif vector[0] < 0 and vector[1] < 0:
                height = (self.gx <= 0) * (self.gy <= 0)
            else:
                if vector[0] == 0:
                    if vector[1] > 0:
                        height = self.gy > 0
                    else:
                        height = self.gy < 0
                if vector[1] == 0:
                    if vector[0] > 0:
                        height = self.gx > 0
                    else:
                        height = self.gx < 0        
 
        # Combining bool arrays
        self.inCylinder = inCylinder * height

        # Getting the velocities of points within the cylinder
        cvx = self.gvx[self.inCylinder] 
        cvy = self.gvy[self.inCylinder]
        cvz = self.gvz[self.inCylinder]

        # Taking away the galaxy veloicty from the gas particles
        cvx = cvx/(1000*100) - self.veloicty[0]
        cvy = cvy/(1000*100) - self.veloicty[1]
        cvz = cvz/(1000*100) - self.veloicty[2]

        # Getting the number of particles in the cylinder
        cylinderParticles = len(cvx)

        if cylinderParticles == 0:
            vlos = "Null"
            weightedMean = "Null"
            cylinderParticles = 0
            
        else:
            # Finding the line of sight velocities
            vels = np.stack([cvx, cvy, cvz], axis=1)
            vlos = np.dot(vels, vectorNorm)

            # Weighted mean velocity
            weightedMean = np.sum(self.gm[self.inCylinder] * vlos) / np.sum(self.gm[self.inCylinder])

        return vlos, weightedMean, cylinderParticles