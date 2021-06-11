import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import jcamp as jc
import matplotlib.pyplot as plt

insert_petrol_coordinates = [[50,320,120], [20,240,60]] #  [[x1,x2, ...], [y1,y2, ...]] 

insert_tnt_coordinates = [[70,350,100], [30,140,30]]

def insert_signal(insert_petrol_coordinates, insert_tnt_coordinates):
    jdx_data_petrol = jc.JCAMP_reader("C4H10.jdx")
    jdx_data_tnt = jc.JCAMP_reader("C7H5N3O6.jdx")
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
        
    # recuperer le signal tnt sur une echelle correspondante a la matrice 3d
    ydata_petrol = jdx_data_petrol['y'][0:len(xdata)]
    ydata_tnt = jdx_data_tnt['y'][0:len(xdata)]
 
    data_dict = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Nov 05 20:20:56 2014', '__version__': '1.0', '__globals__': [], 'Sandiego_before_insert_petrol_tnt': np.array(data3d  , dtype=np.float16)}
    sio.savemat("Sandiego_before_insert_petrol_tnt.mat", data_dict)
    
    """
    Insert a signal at specific location
    """

    """
    Essayer de faire en couleur les signaux 
    """


    for idx in range(len(insert_petrol_coordinates[0])):
        #insert_petrol_coordinates [0][x] == X // insert_petrol_coordinates [1][y] == Y
        data3d[insert_petrol_coordinates[0][idx],insert_petrol_coordinates[1][idx],:] = ydata_petrol

    for idx in range(len(insert_tnt_coordinates[0])):
        #insert_petrol_coordinates [0][x] == X // insert_petrol_coordinates [1][y] == Y
        data3d[insert_tnt_coordinates[0][idx],insert_tnt_coordinates[1][idx],:] = ydata_tnt
           

    data_dict = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Nov 05 20:20:56 2014', '__version__': '1.0', '__globals__': [], 'Sandiego_after_insert_petrol_tnt': np.array(data3d  , dtype=np.float16)}
    sio.savemat("Sandiego_after_insert_petrol_tnt.mat", data_dict)
    




insert_signal([[50,320,120], [20,240,60]]  , [[70,350,100], [30,140,30]])

