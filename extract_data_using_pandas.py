import numpy as np
# import HyperProTool as hyper
import scipy.io as sio
#from LRSR_1 import LRSR
import jcamp as jc

import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection

jdx_data = jc.JCAMP_reader("C7H5N3O6NORMALX.jdx")

# data pre-precessing
data = sio.loadmat("Sandiego.mat")

data3d = np.array(data["Sandiego"], dtype=float)

# normalise data set [0,1]
"""
for idx in range(400):
    norm = np.linalg.norm(data3d[idx][:][:])
    normal_array = data3d[idx][:][:]/norm
    data3d[idx][:][:] = normal_array
"""

xdata = np.linspace(400, 860, num=103)
ydata = data3d[130,120,...].tolist()

#jd_xdata = np.linspace(0, 14304, num=14304)
jd_xdata = jdx_data['x'][0:7140]
jd_ydata = jdx_data['y'][0:7140]
#jd_ydata = jdx_data['y']

# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

ax.plot(jd_xdata[::-1], jd_ydata, color='tab:blue')

# set the limits
#ax.set_xlim([400, 860])
#ax.set_ylim([0, 0.01])

ax.set_xlabel("Longueur d'onde (nm)", fontsize=15)
ax.set_ylabel("Valeur de transmittance", fontsize=15)

ax.set_title("Signature spectrale d'un pixel")

# display the plot
plt.show()
