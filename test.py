import numpy as np
import cv2
import pickle

pickle_in=open("model_trainned.p","rb")
model=pickle.load(pickle_in)

cap=cv2.VideoCapture(0)

while True:
    success,imgOrg=cap.read() 
    img=cv2.cvtColor(imgOrg,cv2.COLOR_BGR2GRAY)
    img=np.asarray(img)
    img=cv2.resize(img,(28,28))
    img=img.reshape(1,28,28,1)
    
    #predict
    classIndex=int(model.predict_classes(img)) #predicts the most likely class
    probVal=np.amax(model.predict(img)) #predicts the probability of the most likely class
    
    if(probVal>0.8):
        cv2.putText(imgOrg,str(classIndex,probVal*100),(50,50),cv2.FONT_HERSHEY_COMPLEX,
        1,(0,0,255),1)
    
    cv2.imshow("processed image",imgOrg)

    if (cv2.waitKey(1)==27):
        break
print(type(img))
print(img.shape)

cv2.destroyAllWindows()
