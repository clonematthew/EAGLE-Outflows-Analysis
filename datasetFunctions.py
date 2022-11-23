# Needed libraries
import h5py
import numpy as np
import astropy.units as u
from pyread_eagle import EagleSnapshot

# Reading Dataset function
def read_dataset(itype, att, nFiles=16):
    # Creating output array
    data = []

    # Loop over each of the files and extract the data
    for i in range(nFiles):
        # Loading in the file
        f = h5py.File('./data/snap_028_z000p000.%i.hdf5'%i, 'r')
        tmp = f['PartType%i/%s'%(itype, att)][...]
        data.append(tmp)

        # Get conversion factors
        cgs = f['PartType%i/%s'%(itype , att )]. attrs.get('CGSConversionFactor')
        aexp = f['PartType%i/%s'%(itype , att )]. attrs.get('aexp-scale-exponent')
        hexp = f['PartType%i/%s'%(itype , att )]. attrs.get('h-scale-exponent')

        # Get expansion factor and Hubble parameter from the header
        a = f['Header'].attrs.get("Time")
        h = f['Header'].attrs.get("HubbleParam")

        # Closing the file
        f.close()

    # Combine all the data into a single array
    if len(tmp.shape) > 1:
        data = np.vstack(data)
    else:
        data = np.concatenate(data)

    # Convert to physical 
    if data.dtype != np.int32 and data.dtype != np.int64:
        data = np.multiply(data, cgs * a ** aexp * h ** hexp, dtype="f8")

    # Returning the data
    return data

# Reading header group function
def read_header():
    # Read in various attributes from the header group
    #f = h5py.File('./data/snap_028_z000p000.0.hdf5', 'r')
    f = h5py.File("/home/universe/spxtd1-shared/RefL0100N1504/snapshot_028_z000p000/snap_028_z000p000.0.hdf5", "r")

    # The Scale Factor
    a = f['Header']. attrs.get('Time')

    # The Hubble Parameter
    h = f['Header']. attrs.get('HubbleParam')

    # The Simulation Box Size
    boxsize = f['Header']. attrs.get('BoxSize') 

    # Closing the file
    f.close ()

    # Returning values
    return a, h, boxsize

# Reading dark matter mass function
def read_dataset_dm_mass():

    # Opening file and getting header information
    f = h5py.File('./data/snap_028_z000p000.0.hdf5', 'r')
    h = f['Header'].attrs.get('HubbleParam')
    a = f['Header'].attrs.get('Time')
    dm_mass = f['Header'].attrs.get('MassTable')[1]
    n_particles = f['Header'].attrs.get('NumPart_Total')[1]

    # Creating an array of length n_particles each with a mass dm_mass
    m = np.ones(n_particles, dtype='f8') * dm_mass

    # Getting the conversion factors
    cgs = f['PartType0/Mass'].attrs.get('CGSConversionFactor')
    aexp = f['PartType0/Mass'].attrs.get('aexp-scale-exponent')
    hexp = f['PartType0/Mass'].attrs.get('h-scale-exponent')

    # Closing the file
    f.close ()

    # Converting to physical units
    m = np.multiply(m, cgs * a**aexp * h**hexp, dtype='f8')

    # Returning the masses
    return m

# Function to read in all particles that are in a galaxy
def read_galaxy(itype, gn, sgn, centre, load_region_length, variables):

    # Allocating data array
    data = {}

    # Getting header info
    a, h, boxsize = read_header()

    # Setting centre to right units
    centre *= h

    # Setting up the read eagle module
    eagle_data = EagleSnapshot("./data/snap_028_z000p000.0.hdf5")

    # Defining the region to load
    region = np.array([
        (centre[0] - 0.5 * load_region_length), (centre[0] + 0.5 * load_region_length),
        (centre[1] - 0.5 * load_region_length), (centre[1] + 0.5 * load_region_length),
        (centre[2] - 0.5 * load_region_length), (centre[2] + 0.5 * load_region_length)
    ])

    # Selecting the region to load
    eagle_data.select_region(*region)

    # Loading in the data using read_eagle
    f = h5py.File("./data/snap_028_z000p000.0.hdf5", "r")

    for att in variables:
        tmp = eagle_data.read_dataset(itype, att)
        cgs =  f["PartType%i/%s" % (itype, att)].attrs.get("CGSConversionFactor")
        aexp = f["PartType%i/%s" % (itype, att)].attrs.get("aexp-scale-exponent")
        hexp = f["PartType%i/%s" % (itype, att)].attrs.get("h-scale-exponent")

        # Periodic wrap coordinates around centre
        if att == "Coordinates": 
            tmp = np.mod(tmp - centre + 0.5 * boxsize, boxsize) + centre - 0.5 * boxsize

        # Converting to proper units   
        data[att] = np.multiply(tmp, cgs * a**aexp * h**hexp, dtype="f8")

    f.close()

    # Mask to selected GroupNumber and SubGroupNumber.
    mask = np.logical_and(data["GroupNumber"] == gn, data["SubGroupNumber"] == sgn)
    for att in data.keys():
        data[att] = data[att][mask]

    return data

# Function to orient galaxy 
def getGalaxyAM(gn, sgn, centre, load_region_length):

    # Attributes we want to load
    attrs = ["GroupNumber", "SubGroupNumber", "Coordinates", "Velocity", "Mass"]
    attrsDM =  ["GroupNumber", "SubGroupNumber", "Coordinates", "Velocity"]

    # Loading in all the galaxy particle data
    gasData = read_galaxy(0, gn, sgn, centre, load_region_length, attrs)
    darkMatterData = read_galaxy(1, gn, sgn, centre, load_region_length, attrsDM)
    starData = read_galaxy(4, gn, sgn, centre, load_region_length, attrs)
    blackHoleData = read_galaxy(5, gn, sgn, centre, load_region_length, attrs)

    # Setting masses of the dark matter particles
    darkMatterData["Mass"] = np.ones(len(darkMatterData["GroupNumber"])) * read_dataset_dm_mass

    # Getting the angular momentum of each component 
    gasAngularMomentum = getAngularMomentumVector(gasData)
    starAngularMomentum = getAngularMomentumVector(starData)
    bhAngularMomentum = getAngularMomentumVector(blackHoleData)
    dmAngularMomentum = getAngularMomentumVector(darkMatterData)

    # Adding them together
    am = gasAngularMomentum + starAngularMomentum + bhAngularMomentum + dmAngularMomentum

    return am / np.linalg.norm(am)

# Function to calculate angular momentum for set of particles
def getAngularMomentumVector(data):

    m = data["Mass"]
    r = data["Coordinates"]
    v = data["Velocity"]

    p = np.cross(r, v)
    l = p * m.reshape(m.size, 1)
    l = np.sum(l, axis=0)

    return l

