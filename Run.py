import cv2
import sys
from time import sleep
import cv2
import tensorflow as tf
import numpy as np
import time
import pickle
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt
import os
x=0
y=0
global im



faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cam = cv2.VideoCapture(0)

while True:
    #sleep(5)
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    #cv2.imshow('Video', img)


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
         cv2.rectangle(img, (x,y), (x+w+50,y+h+50), (255,0,0), 2)
         im = gray[y:y+h,x:x+w]
         cv2.imshow('image',img)
         
         #sleep(3)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.imwrite("main.jpg",im)
        break

cam.release()
cv2.destroyAllWindows()
path = 'train'
CATEGORIES = os.listdir(path)
def prepare(file):
    IMG_SIZE = 50
    img_array = cv2.imread(file, 1)
    #img_array = cv2.Canny(img_array, threshold1=50, threshold2=10)
    img_array = cv2.medianBlur(img_array,1)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    new_array=np.expand_dims(new_array, axis=0)
    return new_array
model = tf.keras.models.load_model("FACE.model")
image = r"main.jpg" #your image path
prediction = model.predict(prepare(image))
prediction = list(prediction[0])
print(prediction)
l2=prediction.index(max(prediction));
print(CATEGORIES[int((l2))])
time.sleep(1)
