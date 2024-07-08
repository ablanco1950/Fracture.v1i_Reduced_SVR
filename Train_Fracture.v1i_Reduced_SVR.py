# Author Alfonso Blanco

import cv2
import time
Ini=time.time()

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

import os
import re

import imutils

import pickle #to save the model

dirname= "trainvalidFractureOJumbo1\images"
dirnameLabels="trainvalidFractureOJumbo1\labels"

########################################################################
def loadimages(dirname):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     TabFileName=[]
   
    
     print("Reading imagenes from ",imgpath)
     NumImage=-2
     
     Cont=0
     for root, dirnames, filenames in os.walk(imgpath):
        
         NumImage=NumImage+1
         
         for filename in filenames:
             
             if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                 
                 
                 filepath = os.path.join(root, filename)
                
                 # SVM only with 2D images
                 #image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                 #print(filepath)
                 #print(image.shape)
                 image = cv2.imread(filepath)
                 image=image.flatten()
                 images.append(image)
                 TabFileName.append(filename)
                 
                 Cont+=1
     
     return images, TabFileName
########################################################################
def loadlabels(dirnameLabels):
 #########################################################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirnameLabels + "\\"
          
    
     TabFileLabelsName=[]
     Yxmidpoint= []
     Yymidpoint= []
     Ywmidpoint= []
     Yhmidpoint= []
     print("Reading labels from ",imgpath)
        
     for root, dirnames, filenames in os.walk(imgpath):
         
         for filename in filenames:
                           
                 filepath = os.path.join(root, filename)
                
                 f=open(filepath,"r")

                 
                 xywh=""
                 for linea in f:                   
                                            
                      xywh=linea[2:]
                      xywh=xywh.split(" ")
                      Yxmidpoint.append(xywh[0])
                      Yymidpoint.append(xywh[1])
                      Ywmidpoint.append(xywh[2])
                      Yhmidpoint.append(xywh[3])
                      
                      break
               
                 TabFileLabelsName.append(filename)
     #print(Yxmidpoint)           
     return  TabFileLabelsName, Yxmidpoint, Yymidpoint, Ywmidpoint, Yhmidpoint



###########################################################
# MAIN
##########################################################

TabFileLabelsName, Yxmidpoint, Yymidpoint, Ywmidpoint, Yhmidpoint= loadlabels(dirnameLabels)

X_train, TabFileName=loadimages(dirname)

print("Number of images to train : " + str(len(TabFileLabelsName)))

#X_train=np.array(X_train)


#print(X_train.shape)


# https://medium.com/@niousha.rf/support-vector-regressor-theory-and-coding-exercise-in-python-ca6a7dfda927

#from sklearn.model_selection import train_test_split

#X_train, X_test = train_test_split(X_train, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler

### When using StandardScaler(), fit() method expects a 2D array-like input
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
#X_test_scaled = scaler.transform(X_test)

from sklearn.svm import SVR

#svr_lin = SVR(kernel = 'linear')
#svr_lin = SVR(kernel = 'poly')

from sklearn.svm import SVC
import pickle #to save the model

from sklearn.multiclass import OneVsRestClassifier


#https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
svr_lin =  OneVsRestClassifier(SVC(kernel='linear', probability=True,  max_iter=1000)) #Creates model instance here
# probar esto
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
#svr_rbf = SVR(kernel = 'rbf')
#svr_poly = SVR(kernel = 'poly')

#print(Yxmidpoint)
svr_lin.fit(X_train_scaled, Yxmidpoint)
#svr_rbf.fit(X_train_scaled, y_train)
#svr_poly.fit(X_train_scaled, y_train)

pickle.dump(svr_lin, open("svr_lin_Yxmidpoint.pickle", 'wb')) #save model as a pickled file

#print(Yxmidpoint)
svr_lin.fit(X_train_scaled, Yymidpoint)
#svr_rbf.fit(X_train_scaled, y_train)
#svr_poly.fit(X_train_scaled, y_train)

pickle.dump(svr_lin, open("svr_lin_Yymidpoint.pickle", 'wb')) #save model as a pickled file

"""
#print(Ywmidpoint)
svr_lin.fit(X_train_scaled, Ywmidpoint)
#svr_rbf.fit(X_train_scaled, y_train)
#svr_poly.fit(X_train_scaled, y_train)

pickle.dump(svr_lin, open("svr_lin_Ywmidpoint.pickle", 'wb')) #save model as a pickled file

print(Yhmidpoint)
svr_lin.fit(X_train_scaled, Yhmidpoint)
#svr_rbf.fit(X_train_scaled, y_train)
#svr_poly.fit(X_train_scaled, y_train)

pickle.dump(svr_lin, open("svr_lin_Yhmidpoint.pickle", 'wb')) #save model as a pickled file

#model2= pickle.load( open("./model.pickle", 'rb'))

"""
