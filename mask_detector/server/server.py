# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 12:10:37 2020

@author: aashu
"""
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os
from PIL import Image
import numpy as np
from flask import send_file
import cv2
import flask
import io

app = flask.Flask(__name__)

def l_model():
    print("[INFO] loading face detector model...")
    prototxtPath = os.path.sep.join([".", "deploy.prototxt"])
    weightsPath = os.path.sep.join([".","res10_300x300_ssd_iter_140000.caffemodel"])
    global net
    net = cv2.dnn.readNet(prototxtPath, weightsPath)
        
    # load the face mask detector model from disk
    print("[INFO] loading face mask detector model...")
    global maskNet
    maskNet = load_model("model_mask")
    
def predict_mask(image):
    #image= cv2.imread(image_path)
    (h,w) = image.shape[:2]
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (300,300),(104.0,177.0,123.0))
    print("[INFO] computing face detections...")
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with
    	# the detection
    	confidence = detections[0, 0, i, 2]
    
    	# filter out weak detections by ensuring the confidence is
    	# greater than the minimum confidence
    	if confidence > 0.5:
    		# compute the (x, y)-coordinates of the bounding box for
    		# the object
    		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    		(startX, startY, endX, endY) = box.astype("int")
    
    		# ensure the bounding boxes fall within the dimensions of
    		# the frame
    		(startX, startY) = (max(0, startX), max(0, startY))
    		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
    
    		# extract the face ROI, convert it from BGR to RGB channel
    		# ordering, resize it to 224x224, and preprocess it
    		face = image[startY:endY, startX:endX]
    		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    		face = cv2.resize(face, (224, 224))
    		face = img_to_array(face)
    		face = preprocess_input(face)
    		face = np.expand_dims(face, axis=0)
    
    		# pass the face through the model to determine if the face
    		# has a mask or not
    		(mask, withoutMask) = maskNet.predict(face)[0]
    
    		# determine the class label and color we'll use to draw
    		# the bounding box and text
    		label = "Mask" if mask > withoutMask else "No Mask"
    		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
    		# include the probability in the label
    		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    	
    		print(label)
    		# display the label and bounding box rectangle on the output
    		# frame
    		print(max(mask,withoutMask)*100)
    		cv2.putText(image, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
    		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
    return image 
    
@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()

            jpg_as_np = np.frombuffer(image, dtype=np.uint8)
            img = cv2.imdecode(jpg_as_np, flags=1)
            img = predict_mask(img)
            #return flask.jsonify("yes")
            img = Image.fromarray(img.astype('uint8'))
            file_object = io.BytesIO()

    # write PNG in file-object
            img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start    
            file_object.seek(0)
            return send_file(file_object, mimetype='image/PNG')   

if __name__ == "__main__":
    print(("* Loading keras model and Fask ..."))
    l_model()
    app.run()