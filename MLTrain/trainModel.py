# This does not work yet.  Its a Work In Progress


import pandas as pd
import numpy as np

from keras.models import Sequential, Model
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers, optimizers
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.optimizers import SGD, Adam



def load_dataFrame():
    trainDf = pd.read_csv(r'.\MLTrain\DataSets\TrainFoodDataSet.csv')  
    testdf = pd.read_csv(r'.\MLTrain\DataSets\TrainFoodDataSet.csv')  

    datagen=ImageDataGenerator(rescale=1./255.,validation_split=0.25)

    train_generator=datagen.flow_from_dataframe(
        dataframe=trainDf,
        directory="./train/",
        x_col="id",
        y_col="label",
        subset="training",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(32,32))

    valid_generator=datagen.flow_from_dataframe(
        dataframe=traindf,
        directory="./train/",
        x_col="id",
        y_col="label",
        subset="validation",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(32,32))




test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(
    dataframe=testdf,
    directory="./test/",
    x_col="id",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(32,32))





HEIGHT = 300
WIDTH = 300

base_model = ResNet50(weights='imagenet', 
                      include_top=False, 
                      input_shape=(HEIGHT, WIDTH, 3))


def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)

    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x) 
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

class_list = ["Pizza", "Burger", "Taco"]
FC_LAYERS = [1024, 1024]
dropout = 0.5

finetune_model = build_finetune_model(base_model, 
                                      dropout=dropout, 
                                      fc_layers=FC_LAYERS, 
                                      num_classes=len(class_list))






NUM_EPOCHS = 10
BATCH_SIZE = 8
num_train_images = 10000

adam = Adam(lr=0.00001)
finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

filepath="./checkpoints/" + "ResNet50" + "_model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]

history = finetune_model.fit_generator(train_generator, epochs=NUM_EPOCHS, workers=8, 
                                       steps_per_epoch=num_train_images // BATCH_SIZE, 
                                       shuffle=True, callbacks=callbacks_list)


!mkdir models
!mkdir models/keras
model.save('models/keras/model.h5')
model = load_model('models/keras/model.h5')

plot_training(history)

# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    # plt.figure()
    # plt.plot(epochs, loss, 'r.')
    # plt.plot(epochs, val_loss, 'r-')
    # plt.title('Training and validation loss')
    plt.show()

    plt.savefig('acc_vs_epochs.png')