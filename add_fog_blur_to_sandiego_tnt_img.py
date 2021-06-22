import numpy as np
# import HyperProTool as hyper
import scipy.io as sio
#from LRSR_1 import LRSR

import matplotlib.pyplot as plt
from matplotlib.collections import EventCollection
import jcamp as jc
from scipy import ndimage, misc
import matplotlib.pyplot as plt
import cv2

from PIL import Image, ImageColor, ImageOps

import random

data = sio.loadmat("Sandiego_after_insert_petrol_tnt.mat")
data3d_brut = np.array(data["Sandiego_after_insert_petrol_tnt"], dtype=float)

def snow_process(image_HLS,snow_point):
    #image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    brightness_coefficient = 3.5 #2.5 
    imshape = image_HLS.shape
    # snow_point=snow_coeff ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
    print(image_HLS[:,:,1])
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    #image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_HLS

def add_snow(image, snow_coeff=-1):
    #verify_image(image)
    if(snow_coeff!=-1):
        if(snow_coeff<0.0 or snow_coeff>1.0):
            raise Exception("err_snow_coeff")
    else:
        snow_coeff=random.uniform(0,1)
    snow_coeff*=0.7/2
    snow_coeff+= 0.7/3 #255/3
        
    output= snow_process(image,snow_coeff)

    return output

def add_fog(image_HLS):
    image_HLS[:,:,1]=image_HLS[:,:,1]*0.8    
    image_HLS[:,:,1] = cv2.blur(image_HLS[:,:,1] ,(10,10), 0)        
    return image_HLS

def apply_mask(mask = add_fog, show = False):
    data = data3d_brut
    # mask ==> function to be applied to matrix layers, by default add_fog 
    for i in range(0,len(data[0,0,:]), 3):
        if show and i< 20:
            cv2.imshow("Result BEFORE MASK {}".format(i), data[:,:,i:i+4])
        data[:,:,i:i+4] = mask(data[:,:,i:i+4])
        if show and i< 20:
            cv2.imshow("Result WITH MASK {}".format(i), data[:,:,i:i+4])
    return data


fogged_data = apply_mask(add_fog)

data_dict = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Nov 05 20:20:56 2014', '__version__': '1.0', '__globals__': [], 'Sandiego_petrol_tnt_with_fog': np.array(fogged_data  , dtype=np.float16)}
sio.savemat("Sandiego_petrol_tnt_with_fog.mat", data_dict)


fogged_data = apply_mask(add_snow)

data_dict = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Nov 05 20:20:56 2014', '__version__': '1.0', '__globals__': [], 'Sandiego_petrol_tnt_with_snow': np.array(fogged_data  , dtype=np.float16)}
sio.savemat("Sandiego_petrol_tnt_with_snow.mat", data_dict)

cv2.waitKey(0)
cv2.destroyAllWindows()