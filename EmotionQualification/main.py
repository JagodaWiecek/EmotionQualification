import cv2
#import imghdr
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.metrics import Precision, Recall, CategoricalAccuracy
import seaborn as sns
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

selected_path = None
hist = None
whatEmotion = None

data = tf.keras.utils.image_dataset_from_directory('data', image_size=(256, 256), batch_size=32)
data_for_testing = tf.keras.utils.image_dataset_from_directory('testing', image_size=(256, 256), batch_size=32)

emotions = ['anger', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

data = data.map(lambda x, y: (x / 255, tf.one_hot(y, depth=8)))
data.as_numpy_iterator().next()

data_for_training = data_for_testing.map(lambda x, y: (x / 255, tf.one_hot(y, depth=8)))
data_for_training.as_numpy_iterator().next()


train_size = int(len(data))
val_size = int(len(data_for_training)*.6)
test_size = int(len(data_for_training)*.4)

print(len(data))
print(len(data_for_training))

train = data.take(train_size)
val = data.take(val_size)
test = data.skip(val_size).take(test_size)

# creating model
model = Sequential()


#model.summary()
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image.thumbnail((300, 300))  # Zmniejszenie rozmiaru obrazu do wyświetlenia w oknie
        photo = ImageTk.PhotoImage(image)
        label.config(image= photo)
        label.image = photo
        global selected_path
        selected_path = file_path

def make_test():
    global selected_path
    global whatEmotion
    if selected_path == None:
        label.config(text="No image, choose some")
    else:
        img = cv2.imread(selected_path)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


def train_model():
    value1 = entry1.get()  # value for epoches
    value2 = entry2.get()  # value for banches
    global val
    global model
    global train
    # adding layers
    if not model.built:
        model.add(Conv2D(16, (3, 3), 1, activation='relu', input_shape=(256, 256, 3)))
        model.add(MaxPooling2D())
        model.add(Conv2D(32, (3, 3), 1, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Conv2D(16, (3, 3), 1, activation='relu'))
        model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(8, activation='softmax'))

        #comlipe model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("model zbudowany")

    label.config(text="Learning... ")
    if value1.isdigit():
        global hist
        if value2.isdigit():
            train = data.take(int(value2))

            logdir = 'logs'
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            hist = model.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])
        else:

            logdir = 'logs'
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
            hist = model.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

    else:
        label.config(text="Integer number obligatory")

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

def MakeMatrix(model):
    global emotions
    global test
    global train
    y_true = np.concatenate([np.argmax(y, axis=-1) for x, y in test], axis=0)
    y_pred_prob = model.predict(test)
    y_pred = np.argmax(y_pred_prob, axis=1)
    assert y_true.shape == y_pred.shape, "y_true and y_pred must have the same shape"
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=emotions, yticklabels=emotions)
    plt.xlabel('Predicted Emotion')
    plt.ylabel('True Emotion')
    plt.title('Confusion Matrix')
    plt.show()

def save_model():
    value = entry3.get()
    global model
    if value is not None:
        model.save(os.path.join('models',str(value) + '.keras'))
    else:
        label.config(text="Enter name for model")


def load_model_from_dialog():

    file_path = filedialog.askopenfilename()
    if file_path is not None:
        global model

        new_model = tf.keras.models.load_model(file_path)
        new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #print(model.built)
        #if not model.built:
         #   print("modelu nie ma")
        model = new_model
        #model.summary()
        #print(model.built)


def evaluate_model():
    pre = Precision()
    re = Recall()
    acc = CategoricalAccuracy()
    #print("deklaracja zmiennych")
    global data
    global test
    #data_iterator = data.as_numpy_iterator()
    #batch = data_iterator.next()
    for batch in test.as_numpy_iterator():
        X, y = batch
        yhat = model.predict(X)
        pre.update_state(y, yhat)
        re.update_state(y, yhat)
        acc.update_state(y, yhat)
    print(f'Precision:{pre.result().numpy()},') #przewidywania
    print(f'Recall: {re.result().numpy()},') #ile przewidywań poprawnie zidentyfikowanych
    print(f'Accuracy: {acc.result().numpy()}') #ile poprawnych przewidywań ogólnie

    MakeMatrix(model)

def manyModelsTest(model):
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

    return model

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
        #Adam compile
        AdamModel.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        AdamHist = AdamModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        SGDModel = Sequential()
        SGDModel = manyModelsTest(SGDModel)
        # Stochastic Gradient Descent (SGD)
        SGDModel.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
        SGDHist = SGDModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        RMSModel = Sequential()
        RMSModel = manyModelsTest(RMSModel)
        # RMSprop
        RMSModel.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
        RMSpropHist = RMSModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        AdagradModel = Sequential()
        AdagradModel = manyModelsTest(AdagradModel)
        # Adagrad
        AdagradModel.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
        AdagradHist = AdagradModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        AdadeltaModel = Sequential()
        AdadeltaModel = manyModelsTest(AdadeltaModel)
        # Adadelta
        AdadeltaModel.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])
        AdadeltaHist = AdadeltaModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        AdamaxModel = Sequential()
        AdamaxModel = manyModelsTest(AdamaxModel)
        # Adamax
        AdamaxModel.compile(optimizer='adamax', loss='categorical_crossentropy', metrics=['accuracy'])
        AdamaxHist = AdamaxModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

        NadamModel = Sequential()
        NadamModel = manyModelsTest(NadamModel)
        # Nadam (Adam with Nesterov momentum)
        NadamModel.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])
        NadamHist = NadamModel.fit(train, epochs=int(value1), validation_data =val, callbacks=[tensorboard_callback])

    else:
        label.config(text="Integer number obligatory")

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
entry1 = tk.Entry(root)
label1 = tk.Label(root, text="Epoches" + ":", padx=5)
label1.pack()
entry1.pack()

label2 = tk.Label(root, text="how many batches(all if null)" + ":", padx=5)
label2.pack()
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
