# Fracture.v1i_Reduced_SVR
Detection of fractures in images by obtaining the X and Y coordinates of the center of the fracture applying ML (SVR). It is applied to a selection of data from the Roboflow file https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/dataset/1 Compared to other tests using DL for the same set of data, much better precision and training time have been obtained

Requirements:

It is required to have at least 36Gb of hard disk space and a computer with a RAM of around 16Gb of RAM is advisable, since the process consumes a lot of these resources.

In case any of the necessary packages are not found, they can all be installed with a simple pip

Functioning.

Two models are obtained separately, one to predict the Xcenter coordinate of the fracture and another to obtain the Ycenter, because SVR only predicts one value for each model. The width and height values ​​of the box could have been obtained by the same procedure, but as indicated, obtaining them consumes memory resources and could give rise to errors due to lack of memory, it is considered sufficient to obtain the position of the center of fracture.

Execution:

download the project and unzip the folders with the trainvalidFractureOJumbo1.zip and testFractureOJumbo1.zip, being careful since some decompressors create double directories and then the programs do not find them (in this case it would be enough to copy the repeated folder and paste it over the project folder)

Train:

Train_Fracture.v1i_Reduced_SVR.py

It takes depending on the computer,approximately 1 hour, but it is simpler than the different training using deep learning that I have used with the same file and cited in the references.

Assessment:

The detection is checked in each of the test images by executing:

Evaluate_Fracture.v1i_Reduced_SVR.py

As the models sizes add up to about 36Gb, there is a time to load then and start the program

In console it presents the list of predicted and true values ​​and then each of the images with a blue circle indicating the point where the fracture has been detected and a green box with which the image was labeled.

A hit rate of 100% is achieved, practically: 9 hits out of 9 images in the test file.

This test is part of a set of tests that use the same reduced set of fracture data applying different DL and ML techniques

- https://github.com/ablanco1950/Fracture.v1i_Reduced_SSD

- https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch

- https://github.com/ablanco1950/Fracture.v1i_Reduced_Yolov10

It outperforms all of them both in accuracy and in much shorter training time.

On the other hand, it uses two models with an exaggerated size: 18Gb and 17Gb, which results in greater memory needs and loading time for the models when establishing the predictions

The results are presented with the 9 test images, green box indicating the labeled box and blue circle indicating the predicted center of the fracture

![Fig1](https://github.com/ablanco1950/Fracture.v1i_Reduced_SVR/blob/main/Figure_1.png)
![Fig2](https://github.com/ablanco1950/Fracture.v1i_Reduced_SVR/blob/main/Figure_2.png)
![Fig3](https://github.com/ablanco1950/Fracture.v1i_Reduced_SVR/blob/main/Figure_3.png)
![Fig4](https://github.com/ablanco1950/Fracture.v1i_Reduced_SVR/blob/main/Figure_4.png)
![Fig5](https://github.com/ablanco1950/Fracture.v1i_Reduced_SVR/blob/main/Figure_5.png)
![Fig6](https://github.com/ablanco1950/Fracture.v1i_Reduced_SVR/blob/main/Figure_6.png)
![Fig7](https://github.com/ablanco1950/Fracture.v1i_Reduced_SVR/blob/main/Figure_7.png)
![Fig8](https://github.com/ablanco1950/Fracture.v1i_Reduced_SVR/blob/main/Figure_8.png)
![Fig9](https://github.com/ablanco1950/Fracture.v1i_Reduced_SVR/blob/main/Figure_9.png)

References:

https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/dataset/1

https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html

https://medium.com/@niousha.rf/support-vector-regressor-theory-and-coding-exercise-in-python-ca6a7dfda927

https://github.com/ablanco1950/Fracture.v1i_Reduced_Yolov10

https://github.com/ablanco1950/Fracture.v1i_Reduced_YoloFromScratch

https://github.com/ablanco1950/Fracture.v1i_Reduced_SSD

https://github.com/ablanco1950/LicenSePlate_Yolov8_FilterSVM_PaddleOCR

https://github.com/ablanco1950/LFW_SVM_facecascade

