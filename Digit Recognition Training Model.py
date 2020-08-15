import urllib  #used for downloading files 
import gzip
import numpy as np
import pickle #used for saving model
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

def load_dataset(): #Download the data and unzip them
    def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
        print('Downloading')
        urllib.urlretrieve(source+filename,filename)
    

    def load_mnsit_images(filename):  #unzipping of image file and converting it into an array
        if not os.path.exists(filename): #Checks if the specified is present in our local disk
        #If not it will download the file
            download(filename)
        with gzip.open(filename,'rb') as f:  #open the zip file of images
            data=np.frombuffer(f.read(), np.unit8, offset=16) #reads the image in the form of 1-d array
            data=data.reshape(-1,28,28,1)  #each image has 1 channel,28*28 pixels,
            #"-1" it indicates that the number of images is going to inferred by other dimensions
            return data/np.float32(255.0) #converting data array values from bytes to float to images

    
    
    def load_mnsit_labels(filename):#unzipping of labels file and converting it into an array
        if not os.path.exists(filename):
            download(filename)
        with gzip.open(filename,'rb') as f:
            data=np.frombuffer(f.read(),np.uint8,offset=8)
            return data
    
    X_train = load_mnsit_images('train-images-idx3-ubyte.gz')
    y_train = load_mnsit_labels('train-labels-idx3-ubyte.gz')
    X_test = load_mnsit_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnsit_labels('t10k-labels-idx3-ubyte.gz')

    y_train = to_categorical(y_train,10) # 10 is number of classes

    return X_train ,y_train ,X_test ,y_test 


# define cnn model on keras
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model



def run_test_harness():  #preparation of conv nn
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# define model
	model = define_model()
	# fit model
	model.fit(trainX, trainY, epochs=10, batch_size=32, verbose=0)
	
run_test_harness()

pickle_out=open(workshop/model_trainned.p","wb")  #workshop is the name of the folder we want to save in
pickle.dump(model,pickle_out)
pickle_out.close()


	

    


