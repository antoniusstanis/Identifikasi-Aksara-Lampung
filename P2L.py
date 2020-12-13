import cv2
import os
import glob
import numpy as np
from matplotlib import pyplot as plt
from skimage import filters
#import thinning

def segment (image):
    #convert BGR image to graycale
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #Binary image
    ret, thresh = cv2.threshold(image_gray, 125, 255, cv2.THRESH_BINARY)

	#Resize image
    scale_percent = 40 #percent of original size
    width = int(thresh.shape[1] * scale_percent / 100)
    height = int(thresh.shape[0] * scale_percent / 100)
    dim = (width, height)

	#resize
    result = cv2.resize(thresh, dim, interpolation=cv2.INTER_AREA)

    #Thinning
    kernel = np.ones((5,5), np.uint8)
    thinned = cv2.dilate(result, kernel, iterations = 3 )
    #opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)

    #thinned = thinning.guo_hall_thinning(result)

    return thinned

    
    #return erosion


if __name__ == '__main__' :
    img_dir = "/media/sena/Data1/SKRIPSICODE/python_mark16/data/dataset/ya"
    data_path = os.path.join(img_dir,'*JPG')
    files = glob.glob(data_path)
    data = []
    result = []

    for f1 in files:
        img = cv2.imread(f1)
        data.append(img)
        result.append(segment(img))

    #save each image
    iteratorName = 1
    prefixPathDir = "/media/sena/Data1/SKRIPSICODE/python_mark16/data/hasil/ya/"
    for img in result :
        cv2.imwrite(prefixPathDir+str(iteratorName)+".png", img)
        iteratorName+=1