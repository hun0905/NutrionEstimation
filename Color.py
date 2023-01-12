import numpy as np
import cv2
def getColorFeature(img):
	
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    h,s,v = cv2.split(img_hsv)
    # hsvHist = [[[0 for _ in range(4)] for _ in range(4)] for _ in range(8)]
    features = []
    hist = cv2.calcHist([img_hsv], [0, 1, 2], None, [8,2,2], [0, 180, 0, 256, 0, 256]).flatten()[1:]
    hist2 = cv2.calcHist([img], [0, 1, 2], None, [4,4,4], [0, 256, 0, 256, 0, 256]).flatten()[1:]
    feature = np.concatenate((hist,hist2),axis=0)
   
    
    # for i in range(12):
    #     for j in range(2):
    #         for k in range(1):
    #             features.append(hist[i][j][k])
    #             features.append(hist2[i][j][k])
    # feature = features[2:]	
    # M = max(feature)
    # m = min(feature)
    # feature = list(map(lambda x: x * 2, feature))
    # feature = (feature - M - m)/(M - m)
    # mean=np.mean(feature)
    # dev=np.std(feature)
    # feature = (feature - mean)/dev
    return np.array(feature)
