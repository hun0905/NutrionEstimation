import cv2
import numpy as np
def getTextureFeature(img):

    #藉由Histograms of Oriented Gradients的方法將影像的紋路特徵提取

    #定義HOG descriptor所需的參數
    winSize = (64,64)
    blockSize = (8,8)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 8
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
    
    #獲取影像的HOG descriptor
    hist = hog.compute(img)
    return hist