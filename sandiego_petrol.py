import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import jcamp as jc
import matplotlib.pyplot as plt

jdx_data = jc.JCAMP_reader("C4H10.jdx")

# data pre-precessing
data = sio.loadmat("Sandiego.mat")

data3d_brut = np.array(data["Sandiego"], dtype=float)
data3d = data3d_brut

# normalise data set [0,1] /MAX
data2d = np.zeros(400*400)
data2d = data2d.reshape(400,400)

for idx in range(400):
    for idy in range(400):
        norm = max(data3d[idx][idy][:])
        data2d[idx,idy] = norm 

for idz in range(224):
    data3d[:,:,idz] = np.abs(data3d[:,:,idz] / data2d) #data3d[x][y][fixed]/data2d[x][y]

xdata = np.linspace(400, 2500, num=224)
xdata  = xdata[xdata < 2500]
xdata  = xdata[xdata > 1000]

data3d = data3d[:,:,0:len(xdata)]

ydata_sandiego = data3d[120,120,...].tolist()

# recuperer le signal tnt sur une echelle correspondante a la matrice 3d
ydata_petrol = jdx_data['y'][0:len(xdata)]

#three locatons ==> left , center and right
X_Sandiego = [120, 200,380]
Y_Sandiego = [120, 200,380]

data_dict = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Nov 05 20:20:56 2014', '__version__': '1.0', '__globals__': [], 'Sandiego_before_insert_petrol': np.array(data3d  , dtype=np.float16)}
sio.savemat("Sandiego_before_insert_petrol.mat", data_dict)

"""
Insert a signal at specific location
"""
for idx in range(len(X_Sandiego)):
    data3d[X_Sandiego[idx],Y_Sandiego[idx],:] = ydata_petrol
    data3d[X_Sandiego[idx],Y_Sandiego[idx],:] = ydata_petrol

data_dict = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Nov 05 20:20:56 2014', '__version__': '1.0', '__globals__': [], 'Sandiego_after_insert_petrol': np.array(data3d  , dtype=np.float16)}

sio.savemat("Sandiego_after_insert_petrol.mat", data_dict)

fig, (ax1,ax2) = plt.subplots(2)

# Using set_dashes() to modify dashing of an existing line
line1, = ax1.plot(xdata, ydata_sandiego, label='Sandiego')

# Using plot(..., dashes=...) to set the dashing when creating a line
line2, = ax2.plot(xdata, ydata_petrol, label='Petrol')

ax1.legend()
ax2.legend()

plt.show()
