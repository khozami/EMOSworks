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
def snow_process(image,snow_coeff):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    brightness_coefficient = 2.5 
    imshape = image.shape
    snow_point=snow_coeff ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2RGB) ## Conversion to RGB
    return image_RGB

def add_snow(image, snow_coeff=-1):
    #verify_image(image)
    if(snow_coeff!=-1):
        if(snow_coeff<0.0 or snow_coeff>1.0):
            raise Exception("err_snow_coeff")
    else:
        snow_coeff=random.uniform(0,1)
    snow_coeff*=255/2
    snow_coeff+=255/3
        
    output= snow_process(image,snow_coeff)
    image_RGB=output

    return image_RGB

def add_fog(image):    
    image_HLS = cv2.cvtColor(image,cv2.COLOR_RGB2HLS) ## Conversion to RGB ==> HLS    
    mask = np.zeros_like(image)
    print("Shape of mask",image.shape)     
    imshape = image.shape
    print("grey scale image shape", image_HLS.shape)    
    hw=100    
    image_HLS[:,:,1]=image_HLS[:,:,1]*0.8    
    
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image_HLS[:,:,1] = cv2.blur(image_HLS[:,:,1] ,(10,10), 0)        
    #image_HLS= add_blur(image_HLS, haze_points[0],haze_points[1], hw) ## adding all shadow polygons on empty mask, single 255 denotes only red channel    
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2BGR) ## Conversion to RGB    
    
    return image_RGB

path = "./image1.jpg"

initial_image = cv2.imread(path)

returned_res = add_snow(initial_image)
returned_res = add_fog(returned_res)

#cv2.imshow("Result", returned_res)
cv2.imwrite("snowandfog.jpg",returned_res)

#cv2.waitKey(0) 
  
#closing all open windows 
#cv2.destroyAllWindows() 


####################################################################################

# data pre-precessing
data = sio.loadmat("Sandiego_after_insert_petrol_tnt.mat")

data3d_brut = np.array(data["Sandiego_after_insert_petrol_tnt"], dtype=float)
# Z == 0

for i in range(0,10, 3):
    print("i==", i, "i+3 == ", i+3)
    im1 = data3d_brut[:,:,i:i+4] # [1,4[
    #im1[:,:,1] = cv2.blur(image_HLS[:,:,1] ,(10,10), 0)
    
    cv2.imshow("Result BEFORE BLUR{}".format(i), im1)
    
    im1[:,:,1] = cv2.blur(im1[:,:,1] ,(10,10), 0)

    cv2.imshow("Result WITH BLUR{}".format(i), im1)
    
    #cv2.imwrite("Result{}.jpg".format(i), im1)

#im1 = data3d_brut[:,:,0] # [1,4[
#im1[:,:,1] = cv2.blur(image_HLS[:,:,1] ,(10,10), 0)

#cv2.imshow("Result{}".format(1), im1)

"""
import numpy as np, cv
vis = np.zeros((384, 836), np.float32)
h,w = vis.shape
vis2 = cv.CreateMat(h, w, cv.CV_32FC3)
vis0 = cv.fromarray(vis)
"""


cv2.waitKey(0)
cv2.destroyAllWindows()