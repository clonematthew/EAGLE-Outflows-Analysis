# Needed libraries
import h5py
import numpy as np

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
    f = h5py.File('./data/snap 028 z000p000 .0.hdf5', 'r')

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
    f = h5py.File('./data/snap 028 z000p000 .0.hdf5', 'r')
    h = f['Header'].attrs.get('HubbleParam')
    a = f['Header'].attrs.get('Time')
    dm_mass = f['Header'].attrs.get('MassTable')[1]
    n_particles = f['Header'].attrs.get('NumPart Total')[1]

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