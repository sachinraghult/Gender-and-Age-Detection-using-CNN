#import libraries
import numpy as np
import pandas as pd
import string
import re
import pickle
import cv2
import os
import shutil
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request


#Initialize the flask App
app = Flask(__name__)


#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')




#face detection and saving cropped images
def detect_face():
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    dirFace = 'static/cropped_face'

    # Create if there is no cropped face directory
    if not os.path.exists(dirFace):
        os.mkdir(dirFace)
        print("Directory " , dirFace ,  " Created ")
    else:    
        shutil.rmtree(dirFace, ignore_errors=True)
        os.mkdir(dirFace)
        print("Directory " , dirFace ,  " Created ")

    path = r'static/file.jpg'

    im = cv2.imread(path, 0)

    # detectfaces 
    faces = classifier.detectMultiScale(
        im, # stream 
        scaleFactor=1.10, # change these parameters to improve your video processing performance
        minNeighbors=20, 
        minSize=(48, 48) # min image detection size
        ) 

    # Draw rectangles around each face
    for (x, y, w, h) in faces:

        cv2.rectangle(im, (x, y), (x + w, y + h),(0,0,255),thickness=2)
        # saving faces according to detected coordinates 
        sub_face = im[y:y+h, x:x+w]
        FaceFileName = "static/cropped_face/face_" + str(y+x) + ".jpg" # folder path and random name image
        cv2.imwrite(FaceFileName, sub_face)
    


def age_group(age):
    if age >=0 and age < 18:
        return 1
    elif age < 30:
        return 2
    elif age < 80:
        return 3
    else:
        return 4 #unknown

def get_age(distr):
    distr = distr*4
    if distr >= 0.65 and distr <= 1.4:return "0-18"
    if distr >= 1.65 and distr <= 2.4:return "19-30"
    if distr >= 2.65 and distr <= 3.4:return "31-80"
    if distr >= 3.65 and distr <= 4.4:return "80 +"
    return "Unknown"
    
def get_gender(prob):
    if prob < 0.5:return "Male"
    else: return "Female"

def get_result(sample, loc):
    sample = sample/255
    model = tf.keras.models.load_model("models/model.h")
    val = model.predict( np.array([ sample ]) )    
    age = get_age(val[0])
    gender = get_gender(val[1])
    res = []
    res.append(loc)
    res.append(age)
    res.append(gender)
    return res



location = []
results = []
images = []


#reduce pixels and writing the data into csv file
def preprocess():
    # assign directory
    directory = 'static/cropped_face'

    folder = os.listdir(directory)
    folder.sort()


    # iterate over files in that directory
    for filename in folder:
        f = os.path.join(directory, filename)
        location.append(f)
        
        # checking if it is a file
        if os.path.isfile(f):
            image = cv2.imread(f, 0)
            img = cv2.resize(image, (64, 64))
            img = img.reshape((64, 64, 1))
            images.append(img)

    x = 0
    for image in images:
        results.append(get_result(image, location[x]))
        x = x + 1




#routing to result page
@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        img = request.files['uploadImage'];
        img.save("static/file.jpg");

        detect_face();
        preprocess();
    
    return render_template('result.html', location=location, results=results)




#turning on debug mode
if __name__ == "__main__":
    app.run(debug=True)
