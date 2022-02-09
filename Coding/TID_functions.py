# Tharsis-Induced Deformation (TID) Model
# This is for Sholes and Rivera-Hernandez (2022) accepted to Icarus
# Please cite both the above and Citron et al. (2018) of which this work
#  is heavily based on. The input data is also provided in this directory
#  which comes from Sholes et al. (2021) in JGR:Planets and their respective
#  original sources.

import numpy as np
import pyshtools as sh
import matplotlib.pyplot as plt
import pandas as pd
import math
import os

# Pixels per degree (used to choose MOLA data)
ppd = 4
# Fixed variables from Citron et al. 2018
sea_citron = -2.308
# Radius of Mars in km
R = 3389.5
# C and sea level used in Citron et al. 2018
citron_params = [1., sea_citron]
citron_ivanov = [0.17, -3.680]

class Shoreline():
    def __init__(self, file_path, C=1.0, lon=1, lat=0, elev=2, region=None, sea_level=None):
        """file_path is a string with the location of the .csv file for the putative shoreline
        C is a float between 0 and 1 showing the percentage of Tharsis formed after the level
        lon is an int showing the csv column where the longitudes are stored
        lat is an int showing the csv column where the latitudes are stored
        elev is an int showing the csv column where the elevation values (in m) are stored
        sea_level is a float dictating what the assumed paleo sea level is (if None, assumes mean paleo elevation)"""

        self.path = file_path
        self.C = C
        self.ppd = ppd

        self.data = np.loadtxt(file_path, skiprows=1, delimiter=',', dtype='float')
        self.data = self.data[self.data[:,lon].argsort()]  #Sorts based on column 1, i.e., Lon

        self.y, self.x = self.data.shape

        if region == None:
            self.df = pd.DataFrame({'LON': self.data[:,lon], 'LAT': self.data[:,lat], 'ELEV':self.data[:,elev]})
        else:
            self.df = pd.DataFrame({'LON': self.data[:,lon], 'LAT': self.data[:,lat], 'ELEV':self.data[:,elev], 'REGION':self.data[:,region]})

        i, TPW_comp, tharsis_comp = 0, np.ones(self.y), np.ones(self.y)
        while i < self.y:
            # This ensures that longitudes are within -180 to 180
            if self.data[i][lon] > 180:
                self.data[i][lon] = self.data[i][lon] - 360

            # This finds the closest map midpoint indices
            m = np.abs(yy - self.data[i][lat]).argmin()
            n = np.abs(xx - self.data[i][lon]).argmin()
            # Finds the corresponding contributions for that cell
            tharsis_comp[i] = tharsis_topo[m][n]
            TPW_comp[i] = topo_TPW[m][n]
            # Add it into the pandas df
            self.df["THARSIS"] = tharsis_comp*self.C
            self.df["TPW"] = TPW_comp*self.C
            self.df["PALEO"] = self.df["ELEV"] - self.df["THARSIS"] - self.df["TPW"]
            i += 1

        self.sea_level = self.df['PALEO'].mean() if sea_level is None else sea_level
        self.df[["ELEV", "THARSIS", "TPW", "PALEO"]] = self.df[["ELEV", "THARSIS", "TPW", "PALEO"]] / 1000.
        self.topo_map = (topo - tharsis_topo*self.C - topo_TPW*self.C) / 1000.

def drop_return(df, index):
    row = df.loc[index]
    df.drop(index, inplace=True)
    return row

def grab_data(path):
    meta_datalist = []

    for root, dirs, files in os.walk(path):
        for file in files:
            # FINDS ALL CSV FILES IN DIRECTORY TO OPEN
            if file.endswith(".csv"):
                print("Opening data from..." + os.path.join(root, file))
                meta_datalist.append(Shoreline(os.path.join(root, file)))
            else:
                continue

    return meta_datalist

def rms(params, df):
    """Root mean square (rms) function that takes an array of parameters:
    percentage of Tharsis development (C) and sea level (Z) along with the data
    to analyze.
    var params: [C, sealevel]"""

    #params the [C, sea_level]
    # this is to allow only passing C if not also passing sea_level:
    C = list(params)[0]
    #print('C: ', C)
    paleotopo = df["ELEV"] - C*df["THARSIS"] - C*df["TPW"]

    # Find the mean paleotopography if not given a fixed sea level to analyze
    # try:
    #     sea_level = list(params)[1]
    # except:
    #     sea_level = paleotopo.mean()

    sea_level = paleotopo.mean()

    # Calculate the RMS
    deform_y = C*df["THARSIS"] + C*df["TPW"] + sea_level
    error = ((df["ELEV"] - deform_y)**2).sum()/(len(df["ELEV"]))

    rms_km = math.sqrt(error)

    return rms_km

def angular_distance(lat,lon,plat,plon):
    '''
    This computes the angular distance from a given latitude and longitude point (lat,lon) to a paleopole location (plat,plon)
    '''
    lon, lat, plon, plat = map(np.radians, [lon, lat, plon, plat])
    gamma = np.arccos(np.sin(lat) * np.sin(plat) \
         + np.cos(lat) * np.cos(plat) * np.cos(lon - plon))
    return np.degrees(gamma)

def P20(x):
    '''computes P_20'''
    p20 = 0.5*(3.*x**2-1)
    return p20

def dT_TPW(lat,lon,plat,plon,hf,onepluskf):
    '''
    This function computes the deformation from True Polar Wander, given
    a current location (lat,lon),
    a paleopole location (lat,lon),
    and fluid love numbers (hf, onepluskf)
    Returns change in topography from TPW (in km)
    '''
    gamma = angular_distance(lat,lon,plat,plon)
    colat = 90.-lat
    omega = 2.*np.pi/(24.6229*3600.) # rot / sec
    a = 3389.5e3 # mean radius in m

    g = 3.71 #gravity of Mars
    dT = 1./3.*omega**2*a**2/1.e3/g*(P20(np.cos(np.radians(gamma)))*(hf - onepluskf)-P20(np.cos(np.radians(colat)))*(hf - onepluskf))
    return dT

def pre_tharsis_maps(ppd=4):
    ########   LOAD THE DATA   ########################
    # Load in the spherical harmonics data
    cslm = np.load('data/shp_cslm.npy')
    CSlm = np.load('data/geo_CSlm.npy')
    # Radius of Mars (in m)
    Rmars = R*1000
    # Load up the MOLA topography
    # I have both 4 ppd (used by Citron et al.) and 32 ppd (slow)
    # MOLA topography is in meters!
    if ppd == 32:
        try:
            topo = np.loadtxt('data/32ppd/mola_32ppd.txt', dtype='float', skiprows=6)
        except:
            print("Go download the MOLA 32 ppd txt file from the PDS")
    elif ppd ==4:
        topo = np.loadtxt('data/megt90n000cb.txt', dtype='float', unpack=True)
    else:
        print("Select a valid pixel per degree (ppd) MOLA value (4 or 32)")
    #Reshape it into appropriate size and reverse image (np.roll)
    topo = np.roll(topo.reshape(ppd*180, ppd*360), ppd*180, axis=1)
    # Define grid spacing
    dx = 1./ppd
    # Mid point spacing of grids
    dx2 = dx/2.
    xx = np.arange(-180+dx2,180,dx) # list of lon-mid points
    yy = np.arange(90.-dx2,-90,-dx) # list of lat mid points
    lons, lats = np.meshgrid(xx,yy,indexing='xy') # 2D mesh of lat/lon mid points

    #############   TRUE POLAR WANDER   #####################
    #Using the fixed paleopole data from Citron et al. 2018
    #The paleopole, according to the best fit of the fossil bulge from Matsuyama and Manga (2010) is given by:
    lat_paleo = 71.1
    lon_paleo = -100.5
    # for elastic lithosphere thickness T_e = 58 km, the h_f and 1+k_f degree-2 love numbers are:
    hf = 2.0
    onepluskf = 2.10

    topo_TPW = dT_TPW(lats, lons ,lat_paleo,lon_paleo,hf,onepluskf)

    #########   THARSIS DEFORMATION ########################
    # Make grids for the Tharsis contribution to the geoid and shape of Mars
    # See Citron et al. (2018)
    tharsis_geoid=sh.expand.MakeGrid2D(CSlm,interval=dx,north=90-dx2,south=-90+dx2,west=-180+dx2,east=180-dx2,norm=3)*Rmars #multiplied by reference radius
    tharsis_shape=sh.expand.MakeGrid2D(cslm,interval=dx,north=90-dx2,south=-90+dx2,west=-180+dx2,east=180-dx2,norm=3)
    tharsis_topo=tharsis_shape-tharsis_geoid
    #topo_sub_tharsis = topo - tharsis_topo

    return topo, tharsis_topo, topo_TPW, xx, yy

def ocean_volume(topo_map, sea_level, ppd=4):
    # Make an array that has the depth where the ocean is and 0 elsewhere
    ocean = np.where(topo_map>sea_level, 0, sea_level - topo_map)
    # plt.imshow(ocean, extent=(-180,180,-90,90), cmap='copper')
    # plt.show()

    dx = 1/ppd # degress per pixel
    dx2 = dx / 2. # grid spacing
    lats = np.arange(90.-dx2,-90,-dx) # list of lat mid points
    # Approximate the area for each latitude bin:
    area = (R*np.cos(np.abs(np.radians(lats)))*np.pi*2/(ppd*360.))*(2.*np.pi*R/(ppd*360.))
    # Extend it to a full 2D grid (and transpose it)
    area = np.array([area,]*360*ppd).T

    # Calculate the volume
    volume = np.sum(np.multiply(area, ocean))

    SA = 1.44e8 #km2
    GEL = (volume / SA) *1000  # convert to GEL in meters
    return volume, GEL

def print_stats(level, section=None, name=None, paleo='PALEO', Ci=None):
    """Prints out the Table Stats for the given Shoreline (defined by class)"""
    # This is just to help with printing stats for sectioned data
    df = level.df if section is None else section
    sea_level = level.sea_level if section is None else section[paleo].mean()
    C = level.C if Ci is None else Ci

    if section is not None:
        topo_map = (topo - C*tharsis_topo - C*topo_TPW) / 1000.
    else:
        topo_map = level.topo_map

    volume, GEL = ocean_volume(topo_map, sea_level, level.ppd)

    print(f'\nStats for {name}')
    print(f'Mean Modern Elevation: {df["ELEV"].mean():.2f}')
    print(f'Modern StD: {df["ELEV"].std():.2f}')
    print(f'Modern Range: {df["ELEV"].max()-df["ELEV"].min():.2f}')
    print(f'Mean Paleo Elevation: {df[paleo].mean():.2f}')
    print(f'Paleo StD: {df[paleo].std():.2f}')
    print(f'Paleo Range: {df[paleo].max()-df[paleo].min():.2f}')
    print(f'C: {C:.2f}')
    print(f'Sea Level: {sea_level:.2f}')
    print(f'Ocean Volume: {volume:.2e} km^3  |  GEL: {GEL:.2f} m\n\n')


# Make the base maps for MOLA topography, Tharsis contributions and Tharsis TPW
# Placing it here to quicken things up (rather than within Shoreline class)
# xx and yy are the grid midpoint coordinates
topo, tharsis_topo, topo_TPW, xx, yy = pre_tharsis_maps(ppd=ppd)

# Excess topography map from Bouley et al. 2016:
bouley = np.loadtxt('data/Bouley2016_FlattenedNorthernPlains.txt', dtype='float', skiprows=1)
bouley = np.flipud(np.roll(bouley.reshape(360,180).T, 180, axis=1))
