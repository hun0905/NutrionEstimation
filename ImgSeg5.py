import cv2
import numpy as np
def color_histeq(img):
    hist_b=cv2.calcHist(img,[0],None,histSize=[256],ranges=[0,256])[:,0]
    hist_g=cv2.calcHist(img,[1],None,histSize=[256],ranges=[0,256])[:,0]
    hist_r=cv2.calcHist(img,[2],None,histSize=[256],ranges=[0,256])[:,0]
    # np.cumsum(hist_b)
    cdf_b=np.cumsum(hist_b)
    cdf_g=np.cumsum(hist_g)
    cdf_r=np.cumsum(hist_r)


    out_b=255*(cdf_b-np.min(cdf_b[cdf_b>0]))/(np.max(cdf_b[cdf_b>0])-np.min(cdf_b[cdf_b>0]))
    out_b[out_b<0]=0
    out_b[out_b>255]=255
    out_b = np.uint8(out_b)
    out_g=255*(cdf_g-np.min(cdf_g[cdf_g>0]))/(np.max(cdf_g[cdf_g>0])-np.min(cdf_g[cdf_g>0]))
    out_g[out_g<0]=0
    out_g[out_g>255]=255
    out_g = np.uint8(out_g)
    out_r=255*(cdf_r-np.min(cdf_r[cdf_r>0]))/(np.max(cdf_r[cdf_r>0])-np.min(cdf_r[cdf_r>0]))
    out_r[out_r<0]=0
    out_r[out_r>255]=255
    out_r = np.uint8(out_r)
    out = cv2.merge((out_b[img[:,:,0]], out_g[img[:,:,1]], out_r[img[:,:,2]]))
    return out
    
def getFood(img):
    #將影像進行濾波並找出其輪廓
    hsv_img = cv2.cvtColor(img ,cv2.COLOR_BGR2HSV)
    if np.mean(hsv_img[:,:,2])<60:
        img = color_histeq(img)

    img_filt=cv2.pyrMeanShiftFiltering(img,20,50)
    img_filt = cv2.cvtColor(img_filt, cv2.COLOR_BGR2GRAY)
    img_th = cv2.adaptiveThreshold(img_filt,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, _ = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #找出整個盤子的輪廓(以面積來看，太大或小不可能是盤子)
    mask = np.zeros(img_filt.shape, np.uint8)
    contours = sorted(contours, key=cv2.contourArea)
    i=-1
    NotPlate = True
    #先把輪廓排序，然後以下是從最大面積的輪廓開始我回找，找到面積符合者輪廓的index
    while cv2.contourArea(contours[i]) > 200000 or NotPlate:
        mask2 = mask.copy()
        cv2.drawContours(mask2, [contours[i]], 0, (255,255,255,255), -1)
        center=np.round(np.mean(cv2.boxPoints(cv2.minAreaRect(contours[i])),axis=0))
        if mask2[int(center[0]),int(center[1])]== 255 and cv2.contourArea(contours[i]) < 200000:
            NotPlate=False
        else:
            i-=1
            NotPlate=True
        if cv2.contourArea(contours[i]) < 100000:
            NotPlate=False
    if cv2.contourArea(contours[i]) < 100000:
        print('No plate detected')
        mask2 = np.ones(img_filt.shape, np.uint8)*255
    #將盤子輪廓(實心)影像(mask)和原始影像作and運算，得到的結果會是只保留盤子組成的影像
    img_plate = cv2.bitwise_and(img,img,mask = mask2)

    #將影像中的白色組成(盤子)去掉，最後只保留下食物和手指(img_food)
    hsv_img = cv2.cvtColor(img_plate, cv2.COLOR_BGR2HSV)
    mask_plate = cv2.inRange(hsv_img, np.array([0,0,100]), np.array([180,90,255]))
    mask_not_plate = cv2.bitwise_not(mask_plate)
    img_food = cv2.bitwise_and(img_plate,img_plate,mask = mask_not_plate)   
    #把膚色(手指)去掉，最後剩下食物的pixel
    hsv_img = cv2.cvtColor(img_food, cv2.COLOR_BGR2HSV)
    finger = cv2.inRange(hsv_img, np.array([0,10,60]), np.array([10,160,255])) 
    mask = cv2.bitwise_not(finger); #invert finger and black
    food_area = cv2.bitwise_and(img_food,img_food,mask = mask) 

    #將剩食物的影像作影像侵蝕，去掉多餘線條
    img_bin = cv2.inRange( cv2.cvtColor(food_area, cv2.COLOR_BGR2GRAY), 10, 255) 
    img_erode = cv2.erode(img_bin,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations =1)

    #以contour的面積來找，找到大小合理的contour，並判定這些contour是食物的cnotour，將contour外的影像都去除，指保留conotour內的組成，並存到food
    img_th = cv2.adaptiveThreshold(img_erode,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
    contours, _ = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea)
    mask_food = np.zeros(img_bin.shape, np.uint8)
    food = []
    added_food_contour = []
    food_area=[]
    finger2 = finger
    try:
        i=-1
        while cv2.contourArea(contours[i]) > 100000:
            i-=1
        while i>=-len(contours):
            NotAdded = True
            x2, y2, w2, h2 = cv2.boundingRect(contours[i])
            for cnt in added_food_contour:
                x1, y1, w1, h1 = cv2.boundingRect(cnt)
                if np.sqrt(((x1+w1)/2-(x2+w2)/2)**2+((y1+h1)/2-(y2+h2)/2)**2)<50:
                    NotAdded = False
            if cv2.contourArea(contours[i]) >1000 and NotAdded:
                img_tmp = img.copy()
                mask_food = np.zeros(img_bin.shape, np.uint8)
                cv2.drawContours(mask_food, [contours[i]], 0, (255,255,255), -1)
                food_area.append(cv2.contourArea(contours[i]))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))
                mask_food = cv2.dilate(mask_food,kernel,iterations = 2)
                # food.append(cv2.bitwise_and(img_tmp,img,mask = mask_food))
                food.append(img[y2:y2+h2,x2:x2+w2,:])
                added_food_contour.append(contours[i])
            elif cv2.contourArea(contours[i]) <1000:
                break
            i-=1
    except:
        print('No food detected')    

    #找到手指的組成並且計算其面積
    finger2 = finger - mask_food
    #erode before finding contours
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    finger2 = cv2.erode(finger2,kernel,iterations = 1)
    img_th = cv2.adaptiveThreshold(finger2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    contours, _ = cv2.findContours(img_th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(finger.shape, np.uint8)
    contours = sorted(contours, key=cv2.contourArea)
    finger_area =None
    i=-1
    Pixel2cm = 5/40
    try:
        while cv2.contourArea(contours[i]) > 10000:
            i-=1
        cv2.drawContours(mask, [contours[i]], 0, (255,255,255), -1)
        Rect = cv2.minAreaRect(contours[i])
        pts = cv2.boxPoints(Rect)
        mask = np.zeros(finger.shape, np.uint8)
        cv2.drawContours(mask,[np.int0(pts)],0,(255,255,255), -1)
        finger_area = cv2.contourArea(pts)
        # print(max(Rect[1]))
        Pixel2cm = 5.0/max(Rect[1])
    except:
        print('no finger detected') 
    
    return food_area, mask_food, food, finger_area, Pixel2cm
### ood 為各個被分割食物的image所組成的list，例如food[0]表示第一個分割出的食物