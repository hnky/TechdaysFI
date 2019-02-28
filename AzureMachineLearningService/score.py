import keras
from keras.models import model_from_json
import cv2
import json
import numpy as np
import os
import base64
import urllib

from azureml.core.model import Model

def init():
    global loaded_model

    model_root = Model.get_model_path('MyModel')
    
    model_file_json = os.path.join(model_root, 'model.json')
    model_file_h5 = os.path.join(model_root, 'model.h5')
    
    json_file = open(model_file_json, 'r') 
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_file_h5)


def run(raw_data):
    try:
        url = json.loads(raw_data)['url']
        urllib.request.urlretrieve(url, filename="tmp.jpg")
        
        image_data = cv2.imread("tmp.jpg", cv2.IMREAD_GRAYSCALE)
        image_data = cv2.resize(image_data,(96,96))
        image_data = image_data/255
        
        data1=[]
        data1.append(image_data)
        data1 = np.array(data1)
        data1 = data1.reshape((data1.shape)[0],(data1.shape)[1],(data1.shape)[2],1)    

        predicted_labels = loaded_model.predict(data1)

        labels=['dog' if value>0.5 else 'cat' for value in predicted_labels]

        os.remove("tmp.jpg")
        
        return json.dumps(labels)
    except:
        return "error"

    
