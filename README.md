# Fracture.v1i_Reduced_SVR
 Detection of fractures in radiographs by obtaining the X and Y coordinates of the center of the fracture applying ML (SVR) to obtain the values ​​of these coordinates separately. It is applied to a selection of data from the Roboflow file https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/dataset/1 which represents a reduced but homogeneous version of that file.

Requirements:

It is required to have at least 36Gb of hard disk space and a computer with a RAM of around 16Gb of RAM is advisable since the process consumes a lot of these resources.

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

As the models add up to about 36Gb, you have to load them into memory because of the time it takes to boot.

It presents the list of predicted and true values ​​and then each of the images with a blue circle indicating the point where the fracture has been detected and a green box with which the image was labeled.

A hit rate of 100% is achieved, practically: 9 hits out of 9 images in the test file.

References:

https://universe.roboflow.com/landy-aw2jb/fracture-ov5p1/dataset/1

