import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import jcamp 

import numpy as np
# import HyperProTool as hyper
import scipy.io as sio
#from LRSR_1 import LRSR

import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import jcamp as jc

jdx_data = jc.JCAMP_reader("C7H5N3O6NORMALX.jdx")

xdata = np.linspace(1000, 7000, num=7152)
xdata  = xdata[xdata < 2500]
xdata  = xdata[xdata > 1000]

ydata = jdx_data['y'][0:len(xdata)]
# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xdata, ydata, color='tab:blue')

# set the limits
#ax.set_xlim([1000, 2500])
#ax.set_ylim([0, 0.006])

ax.set_xlabel("Longueur d'onde (nm)", fontsize=15)
ax.set_ylabel("Reflectance", fontsize=15)

ax.set_title("Signature spectrale de TNT")

# display the plot
plt.show()

#trying to male mat file from numpy data
# data3d.dtype
#transformed_data3d = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Nov 05 20:20:56 2014', '__version__': '1.0', '__globals__': [], 'Sandiego2': data3d}
#norm = np.linalg.norm(data3d)
#normal_array = data3d/norm
#print("---------------------------TRANSFORMED DATA--------------------------------------")
#print(transformed_data3d)
#print("---------------------------DATA AS IS--------------------------------------")


#data_truth = data3d > 1.0
#print(data3d.shape)
# sio.savemat("Sandiego2.mat", transformed_data3d)
#print(data3d[2:4, 2:4, ...])