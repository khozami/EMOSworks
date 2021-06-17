import matplotlib.pyplot as plt
import numpy as np


# Fixing random state for reproducibility
np.random.seed(19680801)

figsrc, axsrc = plt.subplots()
figzoom, axzoom = plt.subplots()
axsrc.set(xlim=(0, 1), ylim=(0, 1), autoscale_on=False,
          title='Click to zoom')
axzoom.set(xlim=(0.45, 0.55), ylim=(0.4, 0.6), autoscale_on=False,
           title='Zoom window')

x, y, s, c = np.random.rand(4, 200)
s *= 200

axsrc.scatter(x, y, s, c)
axzoom.scatter(x, y, s, c)


def on_press(event):
    if event.button != 1:
        return
    x, y = event.xdata, event.ydata
    axzoom.set_xlim(x - 0.1, x + 0.1)
    axzoom.set_ylim(y - 0.1, y + 0.1)
    figzoom.canvas.draw()

figsrc.canvas.mpl_connect('button_press_event', on_press)
plt.show()




"""
fonction affiche zoom
"""
"""

np.random.seed(19680801)

figsrc, axsrc = plt.subplots()
figzoom, axzoom = plt.subplots()
axsrc.set(xlim=(0, 1), ylim=(0, 1), autoscale_on=False,
          title='Click to zoom')
axzoom.set(xlim=(0.45, 0.55), ylim=(0.4, 0.6), autoscale_on=False,
           title='Zoom window')

x, y, s, c = np.random.rand(4, 200)
s *= 200

axsrc.scatter(xdata, ydata_sandiego)
axzoom.scatter(x, y, s, c)


def on_press(event):
    if event.button != 1:
        return
    x, y = event.xdata, event.ydata
    axzoom.set_xlim(x - 0.1, x + 0.1)
    axzoom.set_ylim(y - 0.1, y + 0.1)
    figzoom.canvas.draw()

figsrc.canvas.mpl_connect('button_press_event', on_press)
plt.show()
"""

"""
FIN zoom
"""


"""
# normalise data set [0,1]
for idx in range(400):
    norm = np.linalg.norm(data3d[idx][:][:])
    normal_array = data3d[idx][:][:]/norm
    data3d[idx][:][:] = normal_array
"""



"""
Save chunks of .mat
"""

"""
start = 0
for idx in range(0,len(xdata), 21):
    #data_dict = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Nov 05 20:20:56 2014', '__version__': '1.0', '__globals__': [], 'new_Sandiego': np.array(data3d[:,:,start:start+20]  , dtype=np.float16)}
    print("idx",idx)
    print("len xdata", len(xdata))
    
"""

#sio.savemat("newSandiego.mat", data_dict)


#ydata = data3d[120,120,...].tolist()


fig, (ax1,ax2) = plt.subplots(2)

# Using set_dashes() to modify dashing of an existing line
line1, = ax1.plot(xdata, ydata_sandiego, label='Sandiego')

# Using plot(..., dashes=...) to set the dashing when creating a line
line2, = ax2.plot(xdata, ydata_tnt, label='TNT')

ax1.legend()
ax2.legend()

#plt.show()

