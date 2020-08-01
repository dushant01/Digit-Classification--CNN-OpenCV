import cv2
import numpy as np
import h5py
import tensorflow as tf
#classifierLoad = tf.keras.models.load_model('model/modeltest.h5')

#model = h5py.File('model.h5', 'r')
model = tf.keras.models.load_model('model.h5')
#print(list(f.keys()))

cap = cv2.VideoCapture(0)
width = 640
height = 480

cap.set(3,width)
cap.set(4,height)

def preProcessing(img) :
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img) #equalise the image, make the lighting of the imag distributed equally
    img = img/255 #scalling
    return img

while True:
    succes, imgOrginal = cap.read()
    img = np.asarray(imgOrginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    #cv2.imshow("Processed img", img)
    img= img.reshape(1,32,32,1)

    #predict
    classIndex = int(model.predict_classes(img))
    predictions = model.predict(img)
    print(predictions)
    probVal = np.amax(predictions)
    print(classIndex ,probVal)

    if probVal> 0.65 :
        cv2.putText(imgOrginal,str(classIndex) + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1,(0,0,255),1)
    cv2.imshow("Original Image", imgOrginal)
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
