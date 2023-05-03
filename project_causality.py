import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

#read in data
file1 = 'evspsbl_Amon_MPI-ESM-LR_esmHistorical_r1i1p1_185001-200512.nc'
file2 = 'sfcWind_Amon_MPI-ESM-LR_historical_r1i1p1_185001-200512.nc'
file3 = 'pr_Amon_MPI-ESM-MR_historical_r1i1p1_185001-200512.nc'

ds1 = Dataset(file1,'r') 
ds2 = Dataset(file2,'r')
ds3 = Dataset(file3,'r')

lon = ds1.variables['lon'][:] 
lat = ds1.variables['lat'][:]
evspsbl = ds1.variables['evspsbl'][:,:,:]
sfcWind = ds2.variables['sfcWind'][:,:,:]
pr = ds3.variables['pr'][:,:,:]

plt.plot(lon)
plt.plot(lat)

# site, Beijing
lon_BJ = 116.3
lat_BJ = 39.9

lon_BJ_index = int(lon_BJ / (360.0/len(lon)))
lat_BJ_index = int(lat_BJ / (180.0/len(lat)) + len(lat)/2)

evspsbl_BJ = evspsbl[:,lat_BJ_index,lon_BJ_index]
sfcWind_BJ = sfcWind[:,lat_BJ_index,lon_BJ_index]
pr_BJ = pr[:,lat_BJ_index,lon_BJ_index]

plt.plot(evspsbl_BJ)
plt.plot(sfcWind_BJ)
plt.plot(pr_BJ)

#q1 = sfcWind_BJ
#q2 = evspsbl_BJ
#q3 = pr_BJ

# definition causality
def calculate_trans_entro(q1,q2):
    # calculate transfer entropy
    Data = np.hstack((q1[:-1].reshape(-1,1), q2[1:].reshape(-1,1), q2[:-1].reshape(-1,1)))
    nBins = 11
    [nData, nSignals] = np.shape(Data)

    # classify data
    binEdges = np.full([nSignals, nBins], np.nan)
    minEdge = np.full([nSignals],np.nan)
    maxEdge = np.full([nSignals],np.nan)

    for s in range(nSignals):
        minEdge[s] = np.min(Data[:,s])
        maxEdge[s] = np.max(Data[:,s])
        binEdges_all = np.linspace(minEdge[s], maxEdge[s], num=nBins+1)
        binEdges[s,0:nBins] = binEdges_all[1:]
    
    classifiedData = np.full([nData,nSignals], np.nan)
    for s in range(nSignals):
        # original dataset
        smat = Data[:,s].copy()
        # define classified matrix
        cmat = np.full([nData], np.nan)
        # loop over local bins
        for e in range(nBins):
            if np.where( smat<= binEdges[s,e])[0].size > 0:
                cmat[ np.where( smat<= binEdges[s,e])[0] ] = e
                smat[ np.where( smat<= binEdges[s,e])[0] ] = maxEdge[s] + 9999
                classifiedData[:,s] = cmat
    
        C = np.full([nBins, nBins, nBins], 0)
        for i in range(nData):
            C[ int(classifiedData[i,0]), int(classifiedData[i,1]), int(classifiedData[i,2])] = C[ int(classifiedData[i,0]), int(classifiedData[i,1]), int(classifiedData[i,2])] + 1

    pXtYwYf = (C+1e-20)/np.sum(C)
    # Marginal PDFs
    pYw=np.sum(np.sum(pXtYwYf,axis=0),axis=1)
    # Joint PDFs
    pXtYw=np.sum(pXtYwYf,axis=2)
    pYwYf=np.sum(pXtYwYf,axis=0)
    
    # transfer Shannon information entropy 
    Shannon=np.sum(pXtYwYf*np.log2(pXtYwYf*pYw/pXtYw/pYwYf))
    #print(Shannon)
    return Shannon

#calculate BJ
entro = np.full([3,3],np.nan)
finaldata = np.hstack((pr_BJ.reshape(-1,1),evspsbl_BJ.reshape(-1,1), sfcWind_BJ.reshape(-1,1)))
for i in range(0,3):
    for j in range(0,3):
        entro[i,j] = calculate_trans_entro(finaldata[:,i],finaldata[:,j])
entropy = pd.DataFrame(entro,index=finaldata.columns,columns=finaldata.columns)

    
    

    
    