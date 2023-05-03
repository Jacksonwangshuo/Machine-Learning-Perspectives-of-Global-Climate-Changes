import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap

#read in data
file1 = 'tas_Amon_MPI-ESM-MR_historical_r1i1p1_185001-200512.nc'
file2 = 'pr_Amon_MPI-ESM-MR_historical_r1i1p1_185001-200512.nc'

ds1 = Dataset(file1,'r')
ds2 = Dataset(file2,'r')

tas = ds1.variables['tas'][:,:,:]
lon = ds1.variables['lon'][:] 
lat = ds1.variables['lat'][:]
pr = ds2.variables['pr'][:,:,:]

plt.plot(lon)
plt.plot(lat)

# site Beijing
lon_BJ = 116.3
lat_BJ = 39.9

lon_BJ_index = int(lon_BJ / (360.0/len(lon)))
lat_BJ_index = int(lat_BJ / (180.0/len(lat)) + len(lat)/2)

tas_BJ = tas[:,lat_BJ_index,lon_BJ_index]
pr_BJ = pr[:,lat_BJ_index,lon_BJ_index]

plt.plot(tas_BJ)
plt.plot(pr_BJ)

X = np.insert(tas_BJ.reshape(-1,1),0,1,axis=1)
y = ((pr_BJ - pr_BJ[0])/pr_BJ[0]).reshape(-1,1)*100 

# B = (XT * X)^-1 * XT * y
B = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.transpose(X)),y)
print(B[1])

# site Singapore
lon_Singapore = 103.85
lat_Singapore = 1.3

lon_Singapore_index = int(lon_Singapore / (360.0/len(lon)))
lat_Singapore_index = int(lat_Singapore / (180.0/len(lat)) + len(lat)/2)

tas_Singapore = tas[:,lat_Singapore_index,lon_Singapore_index]
pr_Singapore = pr[:,lat_Singapore_index,lon_Singapore_index]

X = np.insert(tas_Singapore.reshape(-1,1),0,1,axis=1)
y = ((pr_Singapore - pr_Singapore[0])/pr_Singapore[0]).reshape(-1,1)*100 

B = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.transpose(X)),y)
print(B[1])

# site London
lon_London = 0
lat_London = 52

lon_London_index = int(lon_London / (360.0/len(lon)))
lat_London_index = int(lat_London/ (180.0/len(lat)) + len(lat)/2)

tas_London = tas[:,lat_London_index,lon_London_index]
pr_London = pr[:,lat_London_index,lon_London_index]

X = np.insert(tas_London.reshape(-1,1),0,1,axis=1)
y = ((pr_London - pr_London[0])/pr_London[0]).reshape(-1,1)*100 

B = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.transpose(X)),y)
print(B[1])

# site Rio
lon_Rio = 360-43.2
lat_Rio = 22.9

lon_Rio_index = int(lon_Rio / (360.0/len(lon)))
lat_Rio_index = int(len(lat)/2 - lat_Rio/ (180.0/len(lat)))

tas_Rio = tas[:,lat_Rio_index,lon_Rio_index]
pr_Rio = pr[:,lat_Rio_index,lon_Rio_index]

X = np.insert(tas_Rio.reshape(-1,1),0,1,axis=1)
y = ((pr_Rio - pr_Rio[0])/pr_Rio[0]).reshape(-1,1)*100 

B = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.transpose(X)),y)
print(B[1])

# site, New York
lon_NY = 360-74
lat_NY = 40.7

lon_NY_index = int(lon_NY / (360.0/len(lon)))
lat_NY_index = int(lat_NY / (180.0/len(lat)) + len(lat)/2)

tas_NY = tas[:,lat_NY_index,lon_NY_index]
pr_NY = pr[:,lat_NY_index,lon_NY_index]

X = np.insert(tas_NY.reshape(-1,1),0,1,axis=1)
y = ((pr_NY - pr_NY[0])/pr_NY[0]).reshape(-1,1)*100

B = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.transpose(X)),y)
print(B[1])

# site La
lon_La = 91
lat_La = 29.68

lon_La_index = int(lon_La / (360.0/len(lon)))
lat_La_index = int(lat_La/ (180.0/len(lat)) + len(lat)/2)

tas_La = tas[:,lat_La_index,lon_La_index]
pr_La = pr[:,lat_La_index,lon_La_index]

X = np.insert(tas_La.reshape(-1,1),0,1,axis=1)
y = ((pr_La - pr_La[0])/pr_La[0]).reshape(-1,1)*100 

B = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.transpose(X)),y)
print(B[1])

#pick different time window
#185001-185912 BJ
tas_BJ1 = tas[:121,lat_BJ_index,lon_BJ_index]
pr_BJ1 = pr[:121,lat_BJ_index,lon_BJ_index]

X1 = np.insert(tas_BJ1.reshape(-1,1),0,1,axis=1)
y1 = ((pr_BJ1 - pr_BJ1[0])/pr_BJ1[0]).reshape(-1,1)*100 

B1 = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X1),X1)), np.transpose(X1)),y1)
print(B1[1])

#199601-200512 BJ
tas_BJ2 = tas[1753:1873,lat_BJ_index,lon_BJ_index]
pr_BJ2 = pr[1753:1873,lat_BJ_index,lon_BJ_index]

X2 = np.insert(tas_BJ2.reshape(-1,1),0,1,axis=1)
y2 = ((pr_BJ2 - pr_BJ2[0])/pr_BJ2[0]).reshape(-1,1)*100 

B2 = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X2),X2)), np.transpose(X2)),y2)
print(B2[1])

# global mapping
slope = np.full([96,192], np.nan)
for i in range(96):
    for j in range(192):
        tas_site = tas[:,i,j]
        pr_site = pr[:,i,j]
        
        X = np.insert(tas_site.reshape(-1,1),0,1,axis=1)
        y = ((pr_site - pr_site[0])/pr_site[0]).reshape(-1,1)*100
        
        B = np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(X),X)), np.transpose(X)),y)
        
        slope[i,j] = B[1]

plt.imshow(slope)
plt.clim(-15,15)

lons,lats = np.meshgrid(lon,lat)
map = Basemap(projection='robin', lon_0 = 0)
map.drawcoastlines(linewidth=2)
map.drawmeridians(np.arange(0,360,30))
map.drawparallels(np.arange(-90,90,30))
im = map.pcolormesh(lons, lats, slope, shading = 'flat', latlon=True, cmap=plt.cm.jet)
im.set_clim(-15,15)
cbar = map.colorbar(im,'bottom')









