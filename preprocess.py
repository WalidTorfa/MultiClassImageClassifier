import numpy as np
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from sklearn.metrics import classification_report
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split as tts
from keras.applications.vgg16 import preprocess_input


class image_recognition:
    def __init__(self, data_path, labels):
        self.data_path=data_path
        self.labels=labels
    def getting_data(self):
        Data = []
        classes = []
        file_names = os.listdir(self.data_path)

        for i in (file_names):
            for j in range(len(self.labels)):
                if self.labels[j] in i:
                    classes.append(self.labels[j])
                else:
                    continue
        for i in range(len(file_names)):
            full_path = os.path.join(self.data_path, file_names[(i)])
            Data.append(full_path)

        final_data = pd.DataFrame(list(zip(Data, classes)), columns = ["image_path", "labels"])
        return final_data
    def preprocessing_data(self, dataframe):
        temp = len(self.labels)
        train_data, test_data = tts(dataframe, test_size = 0.2)
        if temp == 2:
            class_mode = "binary"
        else:
            class_mode = "multi_output"
        train_datagen=ImageDataGenerator(rescale=1./255)

        train_datagenerator=train_datagen.flow_from_dataframe(dataframe=train_data,
                                                             x_col="image_path",
                                                             y_col="labels",
                                                             target_size=(150, 150),
                                                             class_mode=class_mode,
                                                             batch_size=32)
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_datagenerator=test_datagen.flow_from_dataframe(dataframe=test_data,
                                                           x_col="image_path",
                                                           y_col="labels",
                                                           target_size=(150, 150),
                                                           class_mode=class_mode,
                                                           batch_size=32)
        return (train_datagenerator, test_datagenerator)
    def build_model(self):

        model = Sequential()
        model.add(Conv2D(64, kernel_size=(3,3), input_shape=(150, 150, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu'))
        model.add(MaxPooling2D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])
        model.summary()
        return model

    def train_model(self,model,TrainGenratorOutput):
        model.fit(TrainGenratorOutput, epochs = 20)
        model_name = "image_model.h5"
        model.save(model_name)
        return model_name


def predictirl(imagename):
    Model = tf.keras.models.load_model("image_model.h5")

    my_image = load_img(imagename, target_size=(150, 150))

    #preprocess the image
    my_image = img_to_array(my_image)
    my_image = my_image.reshape((1, my_image.shape[0], my_image.shape[1], my_image.shape[2]))
    my_image = preprocess_input(my_image)

    #make the prediction
    prediction = Model.predict(my_image)
    return(prediction,np.round(prediction))
