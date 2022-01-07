#import libraries
import numpy as np
import string
import re
import pickle
import cv2
import os
import csv
import shutil
import matplotlib.pyplot as plt
from flask import Flask, render_template, request

#Initialize the flask App
app = Flask(__name__)

#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

def detect_face():
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    dirFace = 'static/cropped_face'

    # Create if there is no cropped face directory
    if not os.path.exists(dirFace):
        os.mkdir(dirFace)
        print("Directory " , dirFace ,  " Created ")
    else:    
        print("Directory " , dirFace ,  " has found.")


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


def reduce_pixels():
    # assign directory
    directory = 'static/cropped_face'

    data = []

    # iterate over files in that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        
        # checking if it is a file
        if os.path.isfile(f):
            image = cv2.imread(f, 0)
            img = cv2.resize(image, (48, 48))
            plt.imshow(img)
            plt.show()

            #extracting pixels from image
            rows,cols = img.shape
            pixels = ""
            
            for i in range(rows):
                for j in range(cols):
                    pixels = pixels + " " + str(img[i,j])

            # data rows of csv file
            data.append([filename, pixels])


    # name of csv file 
    filename = "static/test_data.csv"

    fields = ["img_name", "pixels"]
        
    # writing to csv file 
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
            
        # writing the fields 
        csvwriter.writerow(fields) 
            
        # writing the data rows 
        csvwriter.writerows(data)


    #remove the folder after testing
    


@app.route('/result',methods=['POST'])
def result():
    if request.method == 'POST':
        img = request.files['uploadImage'];
        img.save("static/file.jpg");

        detect_face();
        reduce_pixels();

    return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True)
