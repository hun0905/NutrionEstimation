#%%
import pickle
from Color import getColorFeature
from HOG import getTextureFeature
import numpy as np
import cv2
from ImgSeg5 import getFood
import csv
from Shape import getShapeFeatures
import pprint
def nutrition_dict(food):
    with open(r'density_calorie.csv', 'r') as nutrition:
        reader = csv.DictReader(nutrition)
        for row in reader:
            if row['Food']==food:
                del row['Food']
                val=list(map(float,row.values()))
                val = np.multiply(val[0]*200/100,val)[1:]
                return val
    return np.full((4, 1), False)

def get_nutrition(img,classifier,scaler,classNames,number):
    food_area, mask_food, foods, finger_area, Pixel2cm=getFood(img)
    food_list=[]
    volumn_list=[]
    all_nutrition = []
    for i,food in enumerate(foods):
        # food =  cv2.resize(food, (100, 100))
        if number==0:
            break
        food =  cv2.resize(food, (512, 512))
        features= np.concatenate((getColorFeature(food), getTextureFeature(cv2.resize(food, (64, 64))),getShapeFeatures(cv2.cvtColor(food, cv2.COLOR_BGR2GRAY))), axis=0).reshape(1,-1)
        features_normal = scaler.transform(features)
        prob=classifier.predict_proba(features_normal)
        top5=prob.argsort()[0][-1:-6:-1]
        food_names=[classNames[i] for i in top5]
        food_names.append('self input')
        food_names.append('No food here')
        print(" 此影像為: ",dict(zip([i for i in range(1,8)], food_names)) )
        print()
        cv2.imshow('image', food)
        k=cv2.waitKey(0)
        cv2.destroyAllWindows()
        if k>=49 and k<=53:
            key=food_names[int(k)-49]
            food_list.append(key)
            print(f"\033[92m It is \"{key}\"! \n \033[0m")
            number-=1
        elif k==54:
            k=input('Others? please input: \n')
            key=food_names[int(k)-49]
            food_list.append(key)
            print(f"\033[92m It is \"{key}\"! \n \033[0m")
            number-=1
        else:
            print(f"\033[91m No food in this region!\n \n \033[0m")
            continue

        

        # key=classifier.predict(features_normal)
        # print("最可能是: ",key)
        
    
        #藉由食物和手指的比例來估算影像可能的體積
        # print(food_area[i])
        volumn=(((food_area[i]))**(3/2))*(Pixel2cm**3)
        # print(volumn)
        #獲取單位體積食物對應的營養成分
        val=nutrition_dict(key)
        if not val.all():
            print('NO nutrition data')
        else:
            pprint.pprint(dict(zip(['熱量(大卡)','脂質(g)','碳水化合物(g)','蛋白質(g)'], val)))
            all_nutrition.append(val)
    if number != 0:
        for i in range(number):
            print('Still have food!')
            key=input(dict(zip([i for i in range(1,len(classNames)+1)], classNames))) 
            
            food_list.append(classNames[int(key)-1])
            print(f"\033[92m It is \"{food_list[-1]}\"! \n \033[0m")
            val=nutrtion_dict(food_list[-1])
            if not val.all():
                print('NO nutrition data')
            else:
                pprint.pprint(dict(zip(['熱量(大卡)','脂質(g)','碳水化合物(g)','蛋白質(g)'], val)))
                all_nutrition.append(val)
    total = dict(zip(['熱量(大卡)','脂質(g)','碳水化合物(g)','蛋白質(g)'], np.sum(all_nutrition,axis=0)))
    print('\033[93m \n所有的營養總和(預估值，可能有所誤差):')
    pprint.pprint(total)
if __name__=='__main__':
    img=cv2.imread('11.jpg', cv2.IMREAD_COLOR)


    img =  cv2.resize(img, (512, 512))
    classifier = pickle.load(open(r'FoodClassifier.pkl', 'rb'))
    scaler = pickle.load(open(r'scaler.pkl', 'rb'))
    print('照片中有多少種食物? ')
    cv2.imshow('image', img)
    k=cv2.waitKey(0)
    number = k-48
    get_nutrition(img,classifier,scaler,classifier.classes_,int(number))
