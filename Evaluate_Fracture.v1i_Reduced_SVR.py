# Author Alfonso Blanco

import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np

import os
import re

import pickle #to save, load  the model

dirname= "testFractureOJumbo1\images"
dirnameLabels="testFractureOJumbo1\labels"

########################################################################
def loadimages(dirname):
 #############################[###########################################
 # adapted from:
 #  https://www.aprendemachinelearning.com/clasificacion-de-imagenes-en-python/
 # by Alfonso Blanco García
 ########################################################################  
     imgpath = dirname + "\\"
     
     images = []
     imagesCV=[]
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
                 imagesCV.append(image)
                 image=image.flatten()
                 images.append(image)
                 
                 TabFileName.append(filename)
                 
                 Cont+=1
     
     return imagesCV, images, TabFileName
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
                      Yxmidpoint.append(float(xywh[0]))
                      Yymidpoint.append(float(xywh[1]))
                      Ywmidpoint.append(float(xywh[2]))
                      Yhmidpoint.append(float(xywh[3]))
                      
                      break
               
                 TabFileLabelsName.append(filename)
     #print(Yxmidpoint)           
     return  TabFileLabelsName, Yxmidpoint, Yymidpoint, Ywmidpoint, Yhmidpoint

def plot_image(image, box, boxTrue):
    
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    Cont=0
    #print(box)
    
    upper_left_x_True = boxTrue[0] - boxTrue[2] / 2
    upper_left_y_True = boxTrue[1] - boxTrue[3] / 2
    rect = patches.Rectangle(
            (upper_left_x_True * width, upper_left_y_True * height),
            boxTrue[2] * width,
            boxTrue[3] * height,
            linewidth=2,
            edgecolor="green",
            facecolor="none",
        )
        # Add the patch to the Axes
       
    ax.add_patch(rect)
    """ 
    upper_left_x = box[0] - box[2] / 2
    upper_left_y = box[1] - box[3] / 2
    rect1 = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor="blue",
            facecolor="none",
        )
        # Add the patch to the Axes
    ax.add_patch(rect1)
    """
    plt.show()

###########################################################
# MAIN
##########################################################

TabFileLabelsName, Yxmidpoint, Yymidpoint, Ywmidpoint, Yhmidpoint= loadlabels(dirnameLabels)

imagesCV, X_test, TabFileName=loadimages(dirname)

print("Number of images to test : " + str(len(TabFileLabelsName)))


# https://medium.com/@niousha.rf/support-vector-regressor-theory-and-coding-exercise-in-python-ca6a7dfda927


from sklearn.preprocessing import StandardScaler

### When using StandardScaler(), fit() method expects a 2D array-like input
scaler = StandardScaler().fit(X_test)
X_test_scaled = scaler.transform(X_test)

model_svr_lin_Yxmidpoint= pickle.load( open("svr_lin_Yxmidpoint.pickle", 'rb'))
model_svr_lin_Yymidpoint= pickle.load( open("svr_lin_Yymidpoint.pickle", 'rb'))
#model_svr_lin_Ywmidpoint= pickle.load( open("svr_lin_Ywmidpoint.pickle", 'rb'))
#model_svr_lin_Yhmidpoint= pickle.load( open("svr_lin_Yhmidpoint.pickle", 'rb'))

import numpy as np
from sklearn import metrics

#### Test dataset - metrics ####
y_test_pred_Yxmidpoint = model_svr_lin_Yxmidpoint.predict(X_test_scaled)


print("predicted values for Xcenter:")
print(y_test_pred_Yxmidpoint)
print("true values for Xcenter:")
print(Yxmidpoint)
print("===")

y_test_pred_Yymidpoint = model_svr_lin_Yymidpoint.predict(X_test_scaled)

print("predicted values for Ycenter:")
print(y_test_pred_Yymidpoint)
print("true values for Ycenter:")
print(Yymidpoint)

print("===")
"""
y_test_pred_Ywmidpoint = model_svr_lin_Ywmidpoint.predict(X_test_scaled)

print(y_test_pred_Ywmidpoint)
print("REAL")
print(Ywmidpoint)

print("===")

y_test_pred_Yhmidpoint = model_svr_lin_Yhmidpoint.predict(X_test_scaled)

print(y_test_pred_Yhmidpoint)
print("REAL")
print(Yhmidpoint)
"""
print("==============================================================================")
for i in range (len(imagesCV)):
    img=imagesCV[i]
    height, width, _ = img.shape
    #print(y_test_pred_Yxmidpoint[i])
    #print(width)
    p1=float(y_test_pred_Yxmidpoint[i])* float(width)
    p1=int(p1)
    #print(p1)
    #print(int(p1))
    p2=float(y_test_pred_Yymidpoint[i])* float(height)
    p2=int(p2)
    #print(p2)
    #print(int(p2))
    #print(y_test_pred_Yymidpoint[i])
    #print(height)
    #cv2.circle(img,int(p1),int(p2), 10, (0,255,0), thickness=5)
    cv2.circle(img,(p1,p2), 20, (0,0,255), thickness=5)
    #cv2.imshow("ROI", img)  
    #cv2.waitKey(0)
    """
    # https://stackoverflow.com/questions/9215658/plot-a-circle-with-pyplot
    boxes=[]
    boxes.append(y_test_pred_Yxmidpoint[i])
    boxes.append(y_test_pred_Yymidpoint[i])
    boxes.append(y_test_pred_Ywmidpoint[i])
    boxes.append(y_test_pred_Yhmidpoint[i])

    """
    boxes=[]
    boxesTrue=[]
    boxesTrue.append(Yxmidpoint[i])
    boxesTrue.append(Yymidpoint[i])
    boxesTrue.append(Ywmidpoint[i])
    boxesTrue.append(Yhmidpoint[i])
    #plot_image(imagesCV[i], boxes, boxesTrue)
    plot_image(img, boxes, boxesTrue)
    

