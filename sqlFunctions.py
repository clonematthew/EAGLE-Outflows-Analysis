import eagleSqlTools as sql
import numpy as np

# Defining function
def getGalaxyData(simulation):

    # Connecting to the sql database
    con = sql.connect("krn912", password="cxCEG829")

    # Defining the query
    myQuery =  "SELECT \
                    SH.CentreOfPotential_x as sh_x, \
                    SH.CentreOfPotential_y as sh_y, \
                    SH.CentreOfPotential_z as sh_z,  \
                    SH.GroupNumber as gn, \
                    SH.SubGroupNumber as sgn, \
                    AP.Mass_Star as mass, \
                    SH.Velocity_x as vx, \
                    SH.Velocity_y as vy, \
                    SH.Velocity_z as vz, \
                    AP.SFR as sfr, \
                    SH.GalaxyID as id \
                FROM \
                    %s_SubHalo as SH, \
                    %s_Aperture as AP \
                WHERE \
                    SH.SnapNum = 28 \
                    and SH.GalaxyID = AP.GalaxyID \
                    and AP.ApertureSize = 30 " % (simulation, simulation)

    # Executing the query
    myData = sql.execute_query(con, myQuery)

    # Saving data to txt file
    filename = simulation + "galaxyData.txt"
    np.savetxt(filename, myData, delimiter=",", fmt="%.15f")
