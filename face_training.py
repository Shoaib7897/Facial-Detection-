import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'
'''
LBPH(Local Binary Pattern Histogram) converts image to grayscale.It selects a window of 3*3 pixels which has intensity of each pixel(0-255)
We take central value of the matrix as threshold and all the neighbouring values that are equal or greater are set to 1 else set to 0.
Now the matrix contains only binary values. We need to concatenate each value from each position from the matrix line by line to a new binary value which is now the central value
Then it is converted to decimal and set to central value,this is the LBP procedure.
Histograms are extracted by dividing image int a grid.As images are in grayscale,each histogram will contain values from (0-255)
Suppose we have 8*8 grids then we will have 8*8*256 positions in the final histogram.
Final histogram shows characteristics of the original image.
Face recognition is performed by calculating distance between two histograms (original image histogram and converted image histogram)
Thus,it can represent local features of an image i.e it represents distinct image patches making it robust
'''
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("Cascades/haarcascade_frontalface_default.xml");

# function to get the images and label data
def getImagesAndLabels(path):

    #join() joins two strings and returns a string. os.listdir searches the path so that full path of images in dataset can be stored in imagePaths
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []

    for imagePath in imagePaths:

        PIL_img = Image.open(imagePath).convert('L') # all the images that we got from the imagePaths are converted to grayscale
        img_numpy = np.array(PIL_img,'uint8') 
        '''converts pil image to numpy array tp get a grid of values which are all of the same type. 
        uint8 is used unsigned 8 bit integer. And that is the range of pixel.Therefore, for images uint8 type is used.'''

        id = int(os.path.split(imagePath)[-1].split(".")[1]) #split the path of images by a '.' as seperator and maxsplit is 1 i.e it will seperate the image path into 2 parts only
        faces = detector.detectMultiScale(img_numpy) #detectMultiScale detects faces in the objects

        for (x,y,w,h) in faces: #x=column,y=row,w=width,h=height
            faceSamples.append(img_numpy[y:y+h,x:x+w]) #the images from the dataset are now set as faceSamples
            ids.append(id)

    return faceSamples,ids

print ("\nTraining faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml') # recognizer.save() worked on Mac, but not on Pi

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))