import numpy as np
# import HyperProTool as hyper
import scipy.io as sio # 3.9 vs 2.7
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

err_rain_slant="Numeric value between -20 and 20 is allowed"
err_rain_width="Width value between 1 and 5 is allowed"
err_rain_length="Length value between 0 and 100 is allowed"

def is_numeric(x):
    return type(x) is int

def is_list(x):
    return type(x) is list

def generate_random_lines(imshape,slant,drop_length,rain_type):
    drops=[]
    area=imshape[0]*imshape[1]
    no_of_drops=area//10000

    if rain_type.lower()=='drizzle':
        no_of_drops=area//7700
        drop_length=1
    elif rain_type.lower()=='heavy':
        drop_length=30
    elif rain_type.lower()=='torrential':
        no_of_drops=area//500
        drop_length=60

    for i in range(no_of_drops): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops,drop_length

def rain_process(image_HLS,slant,drop_length,drop_color,drop_width,rain_drops):
    image_t= image_HLS.copy()
    for rain_drop in rain_drops:
        cv2.line(image_t,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    image_HLS= cv2.blur(image_t,(7,7)) ## rainy view are blurry
    brightness_coefficient = 0.7 ## rainy days are usually shady 
    #image_HLS = hls(image) ## Conversion to HLS
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    #image_RGB= rgb(image_HLS,'hls') ## Conversion to RGB
    return image_HLS


def add_rain(image,slant=1,drop_length=1,drop_width=1,drop_color=(200,200,200),rain_type='drizzle'): ## (200,200,200) a shade of gray
    slant_extreme=slant
    if not(is_numeric(slant_extreme) and (slant_extreme>=-20 and slant_extreme<=20)or slant_extreme==-1):
        raise Exception(err_rain_slant)
    if not(is_numeric(drop_width) and drop_width>=0 and drop_width<=5):
        raise Exception(err_rain_width)
    if not(is_numeric(drop_length) and drop_length>=0 and drop_length<=100):
        raise Exception(err_rain_length)

    if(is_list(image)):
        image_RGB=[]
        image_list=image
        imshape = image[0].shape
        if slant_extreme==-1:
            slant= np.random.randint(-5,5) ##generate random slant if no slant value is given
        rain_drops,drop_length= generate_random_lines(imshape,slant,drop_length,rain_type)
        for img in image_list:
            output= rain_process(img,slant_extreme,drop_length,drop_color,drop_width,rain_drops)
            image_RGB.append(output)
    else:
        imshape = image.shape
        if slant_extreme==-1:
            slant= np.random.randint(-5,5) ##generate random slant if no slant value is given
        rain_drops,drop_length= generate_random_lines(imshape,slant,drop_length,rain_type)
        output= rain_process(image,slant_extreme,drop_length,drop_color,drop_width,rain_drops)
        image_RGB=output

    return image_RGB


def snow_process(image_HLS,snow_point):
    #image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    brightness_coefficient = 3.5 #2.5 
    imshape = image_HLS.shape
    # snow_point=snow_coeff ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
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
        if show and i< 10:
            cv2.imshow("Result BEFORE MASK {}".format(i), data[:,:,i:i+4])
        data[:,:,i:i+4] = mask(data[:,:,i:i+4])
        if show and i< 10:
            cv2.imshow("Result WITH MASK {}".format(i), data[:,:,i:i+4])
    return data


fogged_data = apply_mask(add_fog)

data_dict = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Nov 05 20:20:56 2014', '__version__': '1.0', '__globals__': [], 'Sandiego_petrol_tnt_with_fog': np.array(fogged_data  , dtype=np.float16)}
sio.savemat("Sandiego_petrol_tnt_with_fog.mat", data_dict)


snowed_data = apply_mask(add_snow, True)

data_dict = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Nov 05 20:20:56 2014', '__version__': '1.0', '__globals__': [], 'Sandiego_petrol_tnt_with_snow': np.array(snowed_data  , dtype=np.float16)}
sio.savemat("Sandiego_petrol_tnt_with_snow.mat", data_dict)

"""
rained_data = apply_mask(add_rain, True)

data_dict = {'__header__': b'MATLAB 5.0 MAT-file, Platform: PCWIN64, Created on: Wed Nov 05 20:20:56 2014', '__version__': '1.0', '__globals__': [], 'Sandiego_petrol_tnt_with_rain': np.array(rained_data  , dtype=np.float16)}
sio.savemat("Sandiego_petrol_tnt_with_rain.mat", data_dict)
"""

cv2.waitKey(0)
cv2.destroyAllWindows()