import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

#read in data
file1 = 'tas_Amon_MPI-ESM-MR_historical_r1i1p1_185001-200512.nc'
file2 = 'pr_Amon_MPI-ESM-MR_historical_r1i1p1_185001-200512.nc'
file3 = 'evspsbl_Amon_MPI-ESM-LR_esmHistorical_r1i1p1_185001-200512.nc'
file4 = 'sfcWind_Amon_MPI-ESM-LR_historical_r1i1p1_185001-200512.nc'

ds1 = Dataset(file1,'r') 
ds2 = Dataset(file2,'r')
ds3 = Dataset(file3,'r')
ds4 = Dataset(file4,'r')

tas = ds1.variables['tas'][:,:,:]
lon = ds1.variables['lon'][:] 
lat = ds1.variables['lat'][:]
pr = ds2.variables['pr'][:,:,:]
evspsbl = ds3.variables['evspsbl'][:,:,:]
sfcWind = ds4.variables['sfcWind'][:,:,:]

plt.plot(lon)
plt.plot(lat)

# site, Beijing
lon_BJ = 116.3
lat_BJ = 39.9

lon_BJ_index = int(lon_BJ / (360.0/len(lon)))
lat_BJ_index = int(lat_BJ / (180.0/len(lat)) + len(lat)/2)

tas_BJ = tas[:,lat_BJ_index,lon_BJ_index]
pr_BJ = pr[:,lat_BJ_index,lon_BJ_index]
evspsbl_BJ = evspsbl[:,lat_BJ_index,lon_BJ_index]
sfcWind_BJ = sfcWind[:,lat_BJ_index,lon_BJ_index]

plt.plot(tas_BJ)
plt.plot(pr_BJ)
plt.plot(evspsbl_BJ)
plt.plot(sfcWind_BJ)

# prepare data
X = np.hstack((pr_BJ[90:-2].reshape(-1,1), tas_BJ[90:-2].reshape(-1,1),evspsbl_BJ[90:-2].reshape(-1,1), sfcWind_BJ[90:-2].reshape(-1,1)))
y = np.full([len(tas_BJ)-92],np.nan)
for i in range(len(y)):
    y[i] = ( np.mean(tas_BJ[90+i:90+i+3]) - np.mean(tas_BJ[i:i+90])) > 0

X_train = []
y_train = []
for i in range(20,int(len(X)*0.8)):
    X_train.append(X[i-20:i,:])
    y_train.append(y[i])
X_train, y_train = np.array(X_train), np.array(y_train)

X_test = []
y_test = []
for i in range(int(len(X)*0.8), len(X)):
    X_test.append(X[i-20:i,:])
    y_test.append(y[i])
X_test, y_test = np.array(X_test), np.array(y_test)

# build model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

mymodel = Sequential()
mymodel.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
mymodel.add(Dense(units=1, activation = 'sigmoid'))

mymodel.summary()

# compile
mymodel.compile(optimizer='adam', loss='mse')
mymodel.fit(X_train, y_train, epochs = 100, batch_size=30)

# prediction
y_pred = mymodel.predict(X_test)
y_pred = y_pred > 0.5

# compare y_pred, y_test
from sklearn.metrics import confusion_matrix
c = confusion_matrix(y_pred, y_test)
accuracy = (c[0,0]+c[1,1])/(np.sum(c))*100
print(confusion_matrix(y_pred, y_test))