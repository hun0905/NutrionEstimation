import numpy as np
import cv2
def getShapeFeatures(img):
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    moments = cv2.moments(contours[0])
    hu = cv2.HuMoments(moments)
    feature = []
    for i in hu:
        feature.append((i[0]+1e-5)**2)	
 
    return feature