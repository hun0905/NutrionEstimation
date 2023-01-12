from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QTableWidgetItem
from UI import Ui_NutritionHelper
import cv2
import time
import pickle
from ImgSeg5 import getFood
import numpy as np
from Color import getColorFeature
from HOG import getTextureFeature
from Shape import getShapeFeatures
from get_nutrition import nutrition_dict
from PyQt5.QtWidgets import QFileDialog
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
		# in python3, super(Class, self).xxx = super().xxx
        super(MainWindow, self).__init__()
        self.ui = Ui_NutritionHelper()
        self.ui.setupUi(self)
        self.setup_control()
        self.setWindowTitle('Nutrition Helper')
        self.food_num=self.ui.spinBox.value()
        self.classifier = pickle.load(open(r'FoodClassifier.pkl', 'rb'))
        self.scaler = pickle.load(open(r'scaler.pkl', 'rb'))
        self.classNames=self.classifier.classes_
        self.hide_checkbox(True)
        self.food_names=None
        self.found_food=0
        self.food_list=[]
        self.start=True
        self.ui.tableWidget.setHidden(True)
        self.ui.label_still.setHidden(True)
        
    def setup_control(self):
        self.img_path = 'plate.jpg'
        # self.img = cv2.imread(self.img_path, cv2.IMREAD_COLOR)
        self.img = cv2.imdecode(np.fromfile(self.img_path,dtype=np.uint8),-1)
        self.img=cv2.resize(self.img, (512, 512))
        food_area, mask_food, self.food, self.finger_area, Pixel2cm=getFood(self.img)
        self.display_img(self.img)
        self.set_food_number()
        self.ui.pushButton.clicked.connect(self.step)
        self.ui.pushButton_file.clicked.connect(self.open_file)
    def display_img(self,img):
        height, width, channel = img.shape
        bytesPerline = 3 * width
        self.qimg = QImage(img, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.label.setPixmap(QPixmap.fromImage(self.qimg))

    def set_food_number(self):
        self.ui.spinBox.valueChanged.connect(self.valuechange)
        self.ui.spinBox.setRange(1,10)
    def valuechange(self):
        self.food_num = self.ui.spinBox.value()
    def step(self):
        if self.found_food<=self.food_num:
            if self.check_result() and self.start==False:
                self.found_food+=1
                self.food_list.append(self.check_result())
            self.start=False
        if self.found_food < self.food_num:
            if self.found_food == self.food_num-1:
                self.ui.pushButton.setText('evaluate')
            else:
                self.ui.pushButton.setText('next picture')
            if self.found_food < len(self.food):
                food=cv2.resize(self.food[self.found_food], (512, 512))
                self.display_img(food)
                features= np.concatenate((getColorFeature(food), getTextureFeature(cv2.resize(self.food[self.found_food], (64, 64))),getShapeFeatures(cv2.cvtColor(food, cv2.COLOR_BGR2GRAY))), axis=0).reshape(1,-1)
                features_normal = self.scaler.transform(features)
                prob=self.classifier.predict_proba(features_normal)
                top5=prob.argsort()[0][-1:-6:-1]
                self.food_names=[self.classNames[i] for i in top5]
                self.food_names.append('self input')
                self.food_names.append('No food here')
                self.set_checkbox(self.food_names)
            else:
                self.hide_checkbox(True)
                self.ui.checkBox_7.setHidden(False)
                self.ui.checkBox_7.setChecked(True)
                self.display_img(self.img)
                self.ui.label_still.setHidden(False)
                self.ui.label_still.setText(f'Still have food : {self.food_num-self.found_food}')
        elif self.found_food == self.food_num:
            self.ui.pushButton.setText('Exit')
            self.found_food+=1
            self.hide_checkbox(True)
            self.display_img(self.img)
            self.evaluate()
            self.ui.label_still.setHidden(True)
        elif self.found_food > self.food_num:
            self.found_food=0
            self.ui.pushButton.setText('Start')
            self.ui.tableWidget.setHidden(True)
            self.food_list=[]
            self.start=True
            
    def set_checkbox(self,food_names):
        self.hide_checkbox(False)
        self.ui.checkBox_1.setText(food_names[0])
        self.ui.checkBox_2.setText(food_names[1])
        self.ui.checkBox_3.setText(food_names[2])
        self.ui.checkBox_4.setText(food_names[3])
        self.ui.checkBox_5.setText(food_names[4])
        self.ui.checkBox_6.setText('No Food')
        self.ui.checkBox_7.setText('Others')
    def check_result(self):
        if self.ui.checkBox_1.isChecked():
            return self.food_names[0]
        if self.ui.checkBox_2.isChecked():
            return self.food_names[1]
        if self.ui.checkBox_3.isChecked():
            return self.food_names[2]
        if self.ui.checkBox_4.isChecked():
            return self.food_names[3]
        if self.ui.checkBox_5.isChecked():
            return self.food_names[4]
        if self.ui.checkBox_6.isChecked():
            return False
        if self.ui.checkBox_7.isChecked():
            return self.ui.lineEdit.text()
    def hide_checkbox(self,hide):
        self.ui.checkBox_1.setHidden(hide)
        self.ui.checkBox_2.setHidden(hide)
        self.ui.checkBox_3.setHidden(hide)
        self.ui.checkBox_4.setHidden(hide)
        self.ui.checkBox_5.setHidden(hide)
        self.ui.checkBox_6.setHidden(hide)
        self.ui.checkBox_7.setHidden(hide)
        self.ui.lineEdit.setHidden(hide)
    def evaluate(self):
        self.ui.tableWidget.setHidden(False)
        self.ui.tableWidget.setColumnCount(self.food_num+2)
        self.ui.tableWidget.setRowCount(5)
        self.ui.tableWidget.setItem(0,0,QTableWidgetItem(str('食物名稱')))
        self.ui.tableWidget.setItem(1,0,QTableWidgetItem(str('熱量(大卡)')))
        self.ui.tableWidget.setItem(2,0,QTableWidgetItem(str('脂質(g)')))
        self.ui.tableWidget.setItem(3,0,QTableWidgetItem(str('碳水化合物(g)')))
        self.ui.tableWidget.setItem(4,0,QTableWidgetItem(str('蛋白質(g)')))
        total=[0,0,0,0]
        for i,food in enumerate(self.food_list):
            if nutrition_dict(food).any()==False:
                n_dict=[0,0,0,0]
                self.ui.tableWidget.setItem(0,i+1,QTableWidgetItem(str('Unknown Food')))
            else:
                n_dict=nutrition_dict(food)
                self.ui.tableWidget.setItem(0,i+1,QTableWidgetItem(str(food)))
            for j,nutrition in enumerate(n_dict):
                self.ui.tableWidget.setItem(j+1,i+1,QTableWidgetItem(str(round(nutrition,2))))
                total[j]+=round(nutrition,2)
        self.ui.tableWidget.setItem(0,self.food_num+1,QTableWidgetItem(str('Total')))
        self.ui.tableWidget.setItem(1,self.food_num+1,QTableWidgetItem(str(total[0])))
        self.ui.tableWidget.setItem(2,self.food_num+1,QTableWidgetItem(str(total[1])))
        self.ui.tableWidget.setItem(3,self.food_num+1,QTableWidgetItem(str(total[2])))
        self.ui.tableWidget.setItem(4,self.food_num+1,QTableWidgetItem(str(total[3])))
    def open_file(self):
        filename, filetype = QFileDialog.getOpenFileName(self,
                  "Open file",
                  "./")      
        try:           
            self.img_path = filename
            print(filename)
            self.img = cv2.imdecode(np.fromfile(self.img_path,dtype=np.uint8),-1)
            self.img=cv2.resize(self.img, (512, 512))
            food_area, mask_food, self.food, self.finger_area, Pixel2cm=getFood(self.img)
            self.display_img(self.img)
        except:
            print('No such file')
     
if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
