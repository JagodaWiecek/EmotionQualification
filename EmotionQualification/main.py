import cv2
#import imghdr
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
import seaborn as sns
import pickle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
 #######TO DO#######
 ##################
#######
data_dir = 'data' #folder where is database
print(os.listdir(data_dir))#all emotions in database

#global variables
selected_path = None
hist = None
whatEmotion = None
emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

data = tf.keras.utils.image_dataset_from_directory('data', image_size=(256, 256), batch_size=32)
data_for_testing = tf.keras.utils.image_dataset_from_directory('testing', image_size=(256, 256), batch_size=32)

data = data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=8)))
data.as_numpy_iterator().next()

data_for_training = data_for_testing.map(lambda x, y: (x / 255, tf.one_hot(y, depth=8)))
data_for_training.as_numpy_iterator().next()

#number of data/batches
train_size = int(len(data))
val_size = int(len(data_for_training)*.6)
test_size = int(len(data_for_training)*.4)

#information how may batches there are
print(len(data))
print(len(data_for_training))
#separating data
train = data.take(train_size)
val = data.take(val_size)
test = data.skip(val_size).take(test_size)

# creating model
model = Sequential()

#function to get image through file dialog
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(image)
        label.config(image= photo)
        label.image = photo
        global selected_path
        selected_path = file_path

#function to put chosen image and check what emotion is on the photo
def make_test():
    global selected_path
    global whatEmotion
    #if there is chosen path
    if selected_path == None:
        label.config(text="No image, choose some")
    else:
        img = cv2.imread(selected_path)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #resize photo to 256x256
        resize = tf.image.resize(img_bgr, (256, 256))
        plt.imshow(resize.numpy().astype(int))
        plt.show()
        yhat = model.predict(np.expand_dims(resize/255, 0))
        predicted_class_index = np.argmax(yhat)
        predicted_emotion = emotions[predicted_class_index]
        print("The person on photo is: ", predicted_emotion)
        whatEmotion = predicted_emotion
        label.config(text=f"The person on photo is: {predicted_emotion} ")
        label4 = tk.Label(root, text=f"The person on photo is: " + str(whatEmotion), padx=5)
        label4.pack()

#function to train created model
def train_model():
    value1 = entry1.get()  # value for epoches
    value2 = entry2.get()  # value for banches
    global val
    global model
    global train
    #if model is not loaded
    if not model.built:
        # adding layers
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(256, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(512, (3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(8, activation='softmax'))
        model.summary()

        #comlipe model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("model zbudowany")


    if value1.isdigit():
        global hist
        if value2.isdigit():
            train = data.take(int(value2))
            #training
            logdir = 'logs'
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            hist = model.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])
        else:
            #training
            logdir = 'logs'
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            hist = model.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

    else:
        label.config(text="Integer number obligatory")
#function to get charts of loss, accuracy, validation loss, validation accuracy
def get_information():

    global hist
    if hist is not None:
        fig = plt.figure()
        plt.plot(hist.history['loss'], color='teal', label='loss')
        plt.plot(hist.history['val_loss'], color='red', label='val_loss')
        fig.suptitle('Loss', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()

        fig = plt.figure()
        plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
        plt.plot(hist.history['val_accuracy'], color='red', label='val_accuracy')
        fig.suptitle('Accuracy', fontsize=20)
        plt.legend(loc="upper left")
        plt.show()
    else:
        print("history is empty, create some history")

#function to save model with .keras in folder "models"
def save_model():
    value = entry3.get()
    global model
    if value is not None:
        modelName = str(value)+'.keras'
        model.save(os.path.join('models', modelName))
    else:
        label.config(text="Enter name for model")

#function to load model from chosen folder
def load_model_from_dialog():

    file_path = filedialog.askopenfilename()
    if file_path is not None:
        global model
        new_model = tf.keras.models.load_model(file_path)
        new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model = new_model

#function to evaluate model,
def evaluate_model():
    pre = Precision()
    re = Recall()
    acc = CategoricalAccuracy()

    global data
    global test
    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(f'Precision:{pre.result().numpy()},') #prediction
    print(f'Recall: {re.result().numpy()},') #how many prediction guessed correctly
    print(f'Accuracy: {acc.result().numpy()}') #how many correct prediction generally

#function with chosen model to test and create charts
def manyModelsTest(model):

    model.add(Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(8, activation='softmax'))

    return model

#function contains seven models with diffrent compile types
#after training four diffrent charts appear
def modelsForDiagram():
    value1 = entry4.get()  # value for epoches
    value2 = entry5.get()  # value for batches
    global val
    global train

    if value2 == "":
        value2 = 600

    AdamHist = None
    SGDHist = None
    RMSpropHist = None
    AdagradHist = None
    AdadeltaHist = None
    AdamaxHist = None
    NadamHist = None

    if value1.isdigit():

        train = data.take(int(value2))
        logdir = 'logs'
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)


        AdamModel = Sequential()
        AdamModel = manyModelsTest(AdamModel)
        AdamModel.summary()
        print("******* New Model 1/7 *******")
        #Adam compile
        AdamModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        AdamHist = AdamModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        print("******* New Model 2/7 *******")
        SGDModel = Sequential()
        SGDModel = manyModelsTest(SGDModel)
        # Stochastic Gradient Descent (SGD)
        SGDModel.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        SGDHist = SGDModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        print("******* New Model 3/7 *******")
        RMSModel = Sequential()
        RMSModel = manyModelsTest(RMSModel)
        # RMSprop
        RMSModel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        RMSpropHist = RMSModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        print("******* New Model 4/7 *******")
        AdagradModel = Sequential()
        AdagradModel = manyModelsTest(AdagradModel)
        # Adagrad
        AdagradModel.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
        AdagradHist = AdagradModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        print("******* New Model 5/7 *******")
        AdadeltaModel = Sequential()
        AdadeltaModel = manyModelsTest(AdadeltaModel)
        # Adadelta
        AdadeltaModel.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        AdadeltaHist = AdadeltaModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        print("******* New Model 6/7 *******")
        AdamaxModel = Sequential()
        AdamaxModel = manyModelsTest(AdamaxModel)
        # Adamax
        AdamaxModel.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
        AdamaxHist = AdamaxModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        print("******* New Model 7/7 *******")
        NadamModel = Sequential()
        NadamModel = manyModelsTest(NadamModel)
        # Nadam (Adam with Nesterov momentum)
        NadamModel.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
        NadamHist = NadamModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

    else:
        label.config(text="Integer number obligatory")
#chart for loss
    fig = plt.figure()
    plt.plot(AdamHist.history['loss'], color='teal', label='Adam-loss')
    plt.plot(SGDHist.history['loss'], color='red', label='SGD-loss')
    plt.plot(RMSpropHist.history['loss'], color='magenta', label='RMS-loss')
    plt.plot(AdagradHist.history['loss'], color='black', label='Adagrad-loss')
    plt.plot(AdadeltaHist.history['loss'], color='purple', label='Adadelta-loss')
    plt.plot(AdamaxHist.history['loss'], color='orange', label='Adamax-loss')
    plt.plot(NadamHist.history['loss'], color='olive', label='Nadam-loss')
    fig.suptitle('Loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
#chart for validation loss
    fig = plt.figure()
    plt.plot(AdamHist.history['val_loss'], color='teal', label='Adam-val_loss')
    plt.plot(SGDHist.history['val_loss'], color='red', label='SGD-val_loss')
    plt.plot(RMSpropHist.history['val_loss'], color='magenta', label='RMS-val_loss')
    plt.plot(AdagradHist.history['val_loss'], color='black', label='Adagrad-val_loss')
    plt.plot(AdadeltaHist.history['val_loss'], color='purple', label='Adadelta-val_loss')
    plt.plot(AdamaxHist.history['val_loss'], color='orange', label='Adamax-val_loss')
    plt.plot(NadamHist.history['val_loss'], color='olive', label='Nadam-val_loss')
    fig.suptitle('validation loss', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
#chart for accuracy
    fig = plt.figure()
    plt.plot(AdamHist.history['accuracy'], color='teal', label='Adam-accuracy')
    plt.plot(SGDHist.history['accuracy'], color='red', label='SGD-accuracy')
    plt.plot(RMSpropHist.history['accuracy'], color='magenta', label='RMS-accuracy')
    plt.plot(AdagradHist.history['accuracy'], color='black', label='Adagrad-accuracy')
    plt.plot(AdadeltaHist.history['accuracy'], color='purple', label='Adadelta-accuracy')
    plt.plot(AdamaxHist.history['accuracy'], color='orange', label='Adamax-accuracy')
    plt.plot(NadamHist.history['accuracy'], color='olive', label='Nadam-accuracy')
    fig.suptitle('Accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()
#chart for validation accuracy
    fig = plt.figure()
    plt.plot(AdamHist.history['val_accuracy'], color='teal', label='Adam-val_accuracy')
    plt.plot(SGDHist.history['val_accuracy'], color='red', label='SGD-val_accuracy')
    plt.plot(RMSpropHist.history['val_accuracy'], color='magenta', label='RMS-val_accuracy')
    plt.plot(AdagradHist.history['val_accuracy'], color='black', label='Adagrad-val_accuracy')
    plt.plot(AdadeltaHist.history['val_accuracy'], color='purple', label='Adadelta-val_accuracy')
    plt.plot(AdamaxHist.history['val_accuracy'], color='orange', label='Adamax-val_accuracy')
    plt.plot(NadamHist.history['val_accuracy'], color='olive', label='Nadam-val_accuracy')
    fig.suptitle('Validation accuracy', fontsize=20)
    plt.legend(loc="upper left")
    plt.show()


#Creating window
root = tk.Tk()
root.geometry("500x700")
root.title("App Emotion qualification")
#zone to write number of epochs for training
entry1 = tk.Entry(root)
label1 = tk.Label(root, text="Epoches" + ":", padx=5)
label1.pack()
entry1.pack()

label2 = tk.Label(root, text="how many batches(all if null)" + ":", padx=5)
label2.pack()
#zone to write number ofbatches for training
entry2 = tk.Entry(root)
entry2.pack()

entry3 = tk.Entry(root)
label3 = tk.Label(root, text="name of saved model" + ":", padx=5)


#Creating buttons
train_button = tk.Button(root, text="Learning", command=train_model)
train_button.pack()

select_button = tk.Button(root, text="Choose testing photo", command=select_image)
select_button.pack()

training_button = tk.Button(root, text="Test", command=make_test)
training_button.pack()

information_button = tk.Button(root, text="Informations", command=get_information)
information_button.pack()

evaluate_button = tk.Button(root, text="Evaluation", command=evaluate_model)
evaluate_button.pack()

SaveModel_button = tk.Button(root, text="Save model", command=save_model)
SaveModel_button.pack()
label3.pack()
entry3.pack()

LoadModel_button = tk.Button(root, text="Load model", command=load_model_from_dialog)
LoadModel_button.pack()


# to create photo
label = tk.Label(root)
label.pack()

entry4 = tk.Entry(root)
label4 = tk.Label(root, text="Epoches for diagram" + ":", padx=5)
label4.pack()
entry4.pack()

entry5 = tk.Entry(root)
label5 = tk.Label(root, text="Batches for diagram(600 deafult)" + ":", padx=5)
label5.pack()
entry5.pack()

#modelsForDiagram()
ChartButton = tk.Button(root, text="Test for charts", command=modelsForDiagram)
ChartButton.pack()


# main loop
root.mainloop()
print("end main loop")
