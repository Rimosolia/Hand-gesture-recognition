from google.colab import drive
drive.mount('/content/drive')
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
samplewise_center=True,
vertical_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory(
        '/content/drive/MyDrive/model_Fi/Data_Hand_Fi/train',
        target_size=(64,64), #Resize phù hợp với mô hình CNN 
        batch_size=32,
        color_mode='grayscale', #Ảnh xám
        class_mode='categorical') #Gán label
test_set = test_datagen.flow_from_directory(
        '/content/drive/MyDrive/model_Fi/Data_Hand_Fi/test',
        target_size=(64,64),#Resize phù hợp với mô hình CNN 
        batch_size=32,
        color_mode='grayscale', #Ảnh xám
        class_mode='categorical')#Gán label
classifier = Sequential()
#step 1 - convolution and polling
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(Activation('relu'))
classifier.add(Convolution2D(32,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(3,3),strides=2))
classifier.add(Dropout(0.25))
#ADDING 2ND CONVOLUTION and polling
classifier.add(Convolution2D(64,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(Activation('relu'))
classifier.add(Convolution2D(64,(3,3),input_shape=(64,64,1),activation='relu',padding='same'))
classifier.add(Activation('relu'))
classifier.add(MaxPooling2D(pool_size=(3,3),strides=2))
classifier.add(Dropout(0.25))
#step3 Flatten
classifier.add(Flatten())
#creating ANN
classifier.add(Dense(units=512,activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(units=5,activation='softmax'))
#complie the CNN
classifier.compile(optimizer='rmsprop',loss='categorical_crossentropy', metrics = ['accuracy'])
#Tiến hành train
classifier.fit_generator(training_set,
        steps_per_epoch=93,#no of images in training set/batch_size
        epochs=3,
        validation_data=test_set,
        validation_steps=46)
#Xuất file .h5 lưu trữ thông số sau khi train
classifier.save('/content/drive/MyDrive/model_Fi/model.h5')
