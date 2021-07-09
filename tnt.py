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

jdx_data = jc.JCAMP_reader("C4H10.jdx")

xdata = np.linspace(1000, 7000, num=7152) # 1 micron --> 7 micron ==> ok

xdata  = xdata[xdata < 2500]
xdata  = xdata[xdata > 1000]

ydata = jdx_data['y'][0:len(xdata)]
# plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(xdata, ydata, color='tab:blue')

ax.set_xlabel("Longueur d'onde (nm)", fontsize=15)
ax.set_ylabel("Reflectance", fontsize=15)

ax.set_title("Signature spectrale de TNT")

# display the plot
plt.show()
