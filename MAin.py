import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
import tensorflow
from keras.models import load_model

path = "myData"

#########################

myList = os.listdir(path)
print("Total number of classes detected", len(myList))

noOfClasses = len(myList)

####we need to store all the images to one list
images = []
Class_no = []
print("Importing Classes ...... ")
for x in range(0,noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    #print(myPicList)
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg= cv2.resize(curImg,(32,32))
        images.append(curImg)
        Class_no.append(x)
    print(x,end= " ")
print(" ")

#### CONVERT TO NUMPY ARRAY
images = np.array(images)
Class_no = np.array(Class_no)


#splitting data

X_train,X_test,Y_train,Y_test = train_test_split(images,Class_no,test_size=0.2)
X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size=0.2)

#print(X_train.shape)
#print(X_test.shape)
#print(X_validation.shape
#print(X_train.shape)

#print(len(np.where(Y_train==0)[0]))

#### PLOT BAR CHART FOR DISTRIBUTION OF IMAGES

numOfSamples= []
for x in range(0, noOfClasses):
    #print(len(np.where(Y_train==x)[0]))
    numOfSamples.append(len(np.where(Y_train == x)[0]))
print(numOfSamples)

plt.figure(figsize=(10, 5))
plt.bar(range(0, noOfClasses), numOfSamples)
plt.title("No. of images for each class")
plt.show()

def preProcessing(img) :
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img) #equalise the image, make the lighting of the imag distributed equally
    img = img/255 #scalling
    return img

#appliying preprocessing
X_train= np.array(list(map(preProcessing, X_train)))
X_test= np.array(list(map(preProcessing, X_test)))
X_validation= np.array(list(map(preProcessing, X_validation)))

#adding depth 1
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

#Working on CNN

dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)

dataGen.fit(X_train)

Y_train = to_categorical(Y_train,noOfClasses)
Y_test = to_categorical(Y_test,noOfClasses)
Y_validation = to_categorical(Y_validation,noOfClasses)

def myModel():
    noOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3 , 3)
    sizeOfPool = (2,2)
    noOfNode = 300

    model = tensorflow.keras.Sequential()
    #tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

    model.add((tensorflow.keras.layers.Conv2D(noOfFilters, sizeOfFilter1,
                                              input_shape=(32,32,1),
                                              activation='relu')))
    model.add((tensorflow.keras.layers.Conv2D(noOfFilters, sizeOfFilter1,activation='relu')))
 #   model.add(tensorflow.keras.layers.MaxPooling2D)
    model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=sizeOfPool))
    model.add((tensorflow.keras.layers.Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add((tensorflow.keras.layers.Conv2D(noOfFilters // 2, sizeOfFilter2, activation='relu')))
    model.add(tensorflow.keras.layers.MaxPooling2D(pool_size=sizeOfPool))
    model.add(tensorflow.keras.layers.Dropout(0.5))

    model.add(tensorflow.keras.layers.Flatten())
    model.add(tensorflow.keras.layers.Dense(noOfNode, activation='relu'))
    model.add(tensorflow.keras.layers.Dropout(0.5))
    model.add(tensorflow.keras.layers.Dense(noOfClasses, activation='softmax'))

    model.compile(tensorflow.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())
batchSizeVal= 50
epochsVal = 10
stepsPerEpochVal = 8

#### STARTING THE TRAINING PROCESS
history = model.fit_generator(dataGen.flow(X_train,Y_train,
                                 batch_size= batchSizeVal),
                                 steps_per_epoch= stepsPerEpochVal,
                                 epochs= epochsVal,
                                 validation_data=(X_validation,Y_validation),
                                 shuffle=1)

#### EVALUATE USING TEST IMAGES
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])

# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")

#### SAVE THE TRAINED MODEL
pickle_out = open("model_trained.pkl", "wb")
pickle.dump(model, pickle_out)
pickle_out.close()

# save model and architecture to single file
model.save("model.h5")
print("Saved model to disk")