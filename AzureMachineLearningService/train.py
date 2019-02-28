import numpy as np
import argparse
import os
import tensorflow as tf
import keras
import cv2
from keras.models import Sequential
from keras.layers import *
from keras.callbacks import *
from keras.optimizers import *
from azureml.core import Run

print("TensorFlow version:", tf.VERSION)

parser = argparse.ArgumentParser()
parser.add_argument('--data-folder', type=str, dest='data_folder', help='data folder mounting point')
args = parser.parse_args()

train_data_path = os.path.join(args.data_folder, 'dataset/train')
print('training dataset is stored here:', train_data_path)

print('Start preparing data')

label=[]
data1=[]
counter=0

for file in os.listdir(train_data_path):
    try:
        image_data=cv2.imread(os.path.join(train_data_path,file), cv2.IMREAD_GRAYSCALE)
        image_data=cv2.resize(image_data,(96,96))
        if file.startswith("cat"):
            label.append(0) #labeling cats pictures with 0
        elif file.startswith("dog"):
            label.append(1) #labeling dogs pictures with 1
        try:
            data1.append(image_data/255)
        except:
            label=label[:len(label)-1]
        counter+=1
        if counter%1000==0:
            print (counter," image data labelled")
    except:
        print ("Failed: ", train_data_path," =>", file)
        

data1 = np.array(data1)
print (data1.shape)
data1 = data1.reshape((data1.shape)[0],(data1.shape)[1],(data1.shape)[2],1)
#data1=data1/255
labels=np.array(label)

print('Done preparing data')

print('Start training')
# start an Azure ML run
run = Run.get_context()


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        run.log('training_acc', np.float(logs.get('acc')))
        run.log('training_loss', np.float(logs.get('loss')))
        self.losses.append(logs.get('acc'))
        
historyLoss = LossHistory()


model=Sequential()
model.add(Conv2D(kernel_size=(3,3),filters=3,input_shape=(96,96,1),activation="relu"))
model.add(Conv2D(kernel_size=(3,3),filters=10,activation="relu",padding="same"))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(kernel_size=(3,3),filters=3,activation="relu"))
model.add(Conv2D(kernel_size=(5,5),filters=5,activation="relu"))
model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
model.add(Conv2D(kernel_size=(2,2),strides=(2,2),filters=10))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(100,activation="sigmoid"))
model.add(Dense(1,activation="sigmoid"))
model.summary()

model.compile(optimizer="adadelta",loss="binary_crossentropy",metrics=["accuracy"])

history = model.fit(data1, labels, validation_split=0.2, epochs=30, batch_size=20, callbacks=[historyLoss])

print('Done training')

print('Start saving')

os.makedirs('./outputs/model', exist_ok=True)
model_json = model.to_json()
with open("./outputs/model/model.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("./outputs/model/model.h5")

print('Done saving')

