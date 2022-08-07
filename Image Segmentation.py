import os
import sys
import numpy as np
import pandas as pd
import re
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
from keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import KFold
import time

# DE-Net
def HCDC(inputTensor, numFilters, kernelSize = 3):
  x = tf.keras.layers.Conv2D(filters = 512, dilation_rate=(1, 1), strides=(1,1),  kernel_size = (3, 3), padding = 'same')(inputTensor)
  x = tf.keras.layers.BatchNormalization()(x)
  x =tf.keras.layers.Activation('relu')(x)
  x2 = tf.keras.layers.Conv2D(filters = 512, dilation_rate=(2, 2), strides=(1,1),  kernel_size = (3, 3), padding = 'same')(x)
  x2 = tf.keras.layers.BatchNormalization()(x2)
  x2 =tf.keras.layers.Activation('relu')(x2)
  x3 = tf.keras.layers.Conv2D(filters = 512, dilation_rate=(5, 5), strides=(1,1),  kernel_size = (3, 3), padding = 'same')(x2)
  x3 = tf.keras.layers.BatchNormalization()(x3)
  x3 =tf.keras.layers.Activation('relu')(x3)
  x4 = tf.keras.layers.Conv2D(filters = 512, dilation_rate=(7, 7), strides=(1,1),  kernel_size = (3, 3), padding = 'same')(x3)
  x4 = tf.keras.layers.BatchNormalization()(x4)
  x4 =tf.keras.layers.Activation('relu')(x4)
  sum = tf.keras.layers.add([x, x2, x3, x4]) # Element-wise summation
  return sum

def Conv2dBlock(inputTensor, numFilters, kernelSize = 3, doBatchNorm = True):
  x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize), kernel_initializer = 'he_normal', padding = 'same') (inputTensor)
  if doBatchNorm: 
    x = tf.keras.layers.BatchNormalization()(x)
  x =tf.keras.layers.Activation('relu')(x)
  return x

# Defining Network 
def GiveMeDEnet(inputImage, numFilters = 64, droupouts = 0.1, doBatchNorm = True):
    # Dilated Feature Encoder
    # 1st Conv Block
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    c1 = Conv2dBlock(c1, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    # 2nd Conv Block
    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    c2 = Conv2dBlock(c2, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    # 3rd Conv Block
    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    c3 = Conv2dBlock(c3, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    c3 = Conv2dBlock(c3, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    # 4th Conv Block
    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    c4 = Conv2dBlock(c4, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    c4 = Conv2dBlock(c4, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    #  HCDC Block
    hcdc = HCDC(c4, numFilters * 8, kernelSize = 3)
    # Feature Decoder
    c6 = tf.keras.layers.Conv2D(filters = numFilters * 4, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(hcdc)
    c6 = tf.keras.layers.Conv2DTranspose(numFilters*4, (4, 4), strides = (1, 1), padding = 'same')(c6)
    c6 = tf.keras.layers.BatchNormalization()(c6)
    c6 = tf.keras.layers.Activation('relu')(c6)
    c6 = tf.keras.layers.concatenate([c6, hcdc])
    c7 = tf.keras.layers.Conv2D(filters = numFilters * 4, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(c6)
    c7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (4, 4), strides = (1, 1), padding = 'same')(c7)
    c7 = tf.keras.layers.BatchNormalization()(c7)
    c7 = tf.keras.layers.Activation('relu')(c7)
    c7 = tf.keras.layers.concatenate([c7, c4])
    c8 = tf.keras.layers.Conv2D(filters = numFilters * 4, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(c7)
    c8 = tf.keras.layers.Conv2DTranspose(numFilters*4, (4, 4), strides = (2, 2), padding = 'same')(c8)
    c8 = tf.keras.layers.BatchNormalization()(c8)
    c8 = tf.keras.layers.Activation('relu')(c8)
    c8 = tf.keras.layers.concatenate([c8, c3])
    c9 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(c8)
    c9 = tf.keras.layers.Conv2DTranspose(64, (4, 4), strides = (2, 2), padding = 'same')(c9)
    c9 = tf.keras.layers.BatchNormalization()(c9)
    c9 = tf.keras.layers.Activation('relu')(c9)
    c9 = tf.keras.layers.concatenate([c9, c2])
    c10 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (1, 1), kernel_initializer = 'he_normal', padding = 'same')(c9)
    c10 = tf.keras.layers.Conv2DTranspose(32, (4, 4), strides = (2, 2), padding = 'same')(c10)
    c10 = tf.keras.layers.BatchNormalization()(c10)
    c10 = tf.keras.layers.Activation('relu')(c10)
    c10 = tf.keras.layers.concatenate([c10, c1])
    output = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(c10)
    model = tf.keras.Model(inputs = [inputImage], outputs = [output])
    return model

# U-Net ORIGINAL

def Conv2dBlock(inputTensor, numFilters, kernelSize = 3, doBatchNorm = True):
    #first Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (inputTensor)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x =tf.keras.layers.Activation('relu')(x)
    #Second Conv
    x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize),
                              kernel_initializer = 'he_normal', padding = 'same') (x)
    if doBatchNorm:
        x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

# Defining Unet 
def GiveMeUnet(inputImage, numFilters = 64, droupouts = 0.1, doBatchNorm = True):
    # defining encoder Path
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    p1 = tf.keras.layers.Dropout(droupouts)(p1)
    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    p2 = tf.keras.layers.Dropout(droupouts)(p2)
    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    p3 = tf.keras.layers.Dropout(droupouts)(p3)
    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4)
    p4 = tf.keras.layers.Dropout(droupouts)(p4)
    #bottom
    c5 = Conv2dBlock(p4, numFilters * 16, kernelSize = 3, doBatchNorm = doBatchNorm)
    # defining decoder path
    u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides = (2, 2), padding = 'same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4]) #
    u6 = tf.keras.layers.Dropout(droupouts)(u6)
    c6 = Conv2dBlock(u6, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    u7 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides = (2, 2), padding = 'same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    u7 = tf.keras.layers.Dropout(droupouts)(u7)
    c7 = Conv2dBlock(u7, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    u8 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides = (2, 2), padding = 'same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    u8 = tf.keras.layers.Dropout(droupouts)(u8)
    c8 = Conv2dBlock(u8, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    u9 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides = (2, 2), padding = 'same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1])
    u9 = tf.keras.layers.Dropout(droupouts)(u9)
    c9 = Conv2dBlock(u9, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    output = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(c9)
    model = tf.keras.Model(inputs = [inputImage], outputs = [output])
    return model

# Unet + VGG16
def Conv2dBlock(inputTensor, numFilters, kernelSize = 3, doBatchNorm = True):
  x = tf.keras.layers.Conv2D(filters = numFilters, kernel_size = (kernelSize, kernelSize), kernel_initializer = 'he_normal', padding = 'same') (inputTensor)
  if doBatchNorm: 
    x = tf.keras.layers.BatchNormalization()(x)
  x =tf.keras.layers.Activation('relu')(x)
  return x

# Now defining Unet 
def GiveMeUnetWithVGG(inputImage, numFilters = 64, droupouts = 0.1, doBatchNorm = True):
    # 1st Conv Block
    c1 = Conv2dBlock(inputImage, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    c1 = Conv2dBlock(c1, numFilters * 1, kernelSize = 3, doBatchNorm = doBatchNorm)
    p1 = tf.keras.layers.MaxPooling2D((2,2))(c1)
    # 2nd Conv Block
    c2 = Conv2dBlock(p1, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    c2 = Conv2dBlock(c2, numFilters * 2, kernelSize = 3, doBatchNorm = doBatchNorm)
    p2 = tf.keras.layers.MaxPooling2D((2,2))(c2)
    # 3rd Conv Block
    c3 = Conv2dBlock(p2, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    c3 = Conv2dBlock(c3, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    c3 = Conv2dBlock(c3, numFilters * 4, kernelSize = 3, doBatchNorm = doBatchNorm)
    p3 = tf.keras.layers.MaxPooling2D((2,2))(c3)
    # 4th Conv Block
    c4 = Conv2dBlock(p3, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    c4 = Conv2dBlock(c4, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    c4 = Conv2dBlock(c4, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    p4 = tf.keras.layers.MaxPooling2D((2,2))(c4) #
    # 5th Conv block
    c5 = Conv2dBlock(p4, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm) #
    c5 = Conv2dBlock(c5, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm) #
    c5 = Conv2dBlock(c5, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm) #
    p5 = tf.keras.layers.MaxPooling2D((2,2))(c5) #
    # Fully connected layers  
    c6 = Conv2dBlock(p5, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm) #
    c6 = Conv2dBlock(c6, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm) #
    # defining decoder path
    u6 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides = (2, 2), padding = 'same')(c6) 
    u6 = tf.keras.layers.Activation('relu')(u6)
    u6 = tf.keras.layers.concatenate([u6, c5])
    u6 = Conv2dBlock(u6, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    u7 = tf.keras.layers.Conv2DTranspose(numFilters*8, (3, 3), strides = (2, 2), padding = 'same')(u6)
    u7 = tf.keras.layers.Activation('relu')(u7)
    u7 = tf.keras.layers.concatenate([u7, c4])
    u7 = Conv2dBlock(u7, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    u8 = tf.keras.layers.Conv2DTranspose(numFilters*4, (3, 3), strides = (2, 2), padding = 'same')(u7)
    u8 = tf.keras.layers.Activation('relu')(u8)
    u8 = tf.keras.layers.concatenate([u8, c3])
    u8 = Conv2dBlock(u8, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    u9 = tf.keras.layers.Conv2DTranspose(numFilters*2, (3, 3), strides = (2, 2), padding = 'same')(u8)
    u9 = tf.keras.layers.Activation('relu')(u9)
    u9 = tf.keras.layers.concatenate([u9, c2])
    u9 = Conv2dBlock(u9, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    u10 = tf.keras.layers.Conv2DTranspose(numFilters*1, (3, 3), strides = (2, 2), padding = 'same')(u9)
    u10 = tf.keras.layers.Activation('relu')(u10)
    u10 = tf.keras.layers.concatenate([u10, c1])
    u10 = Conv2dBlock(u10, numFilters * 8, kernelSize = 3, doBatchNorm = doBatchNorm)
    output = tf.keras.layers.Conv2D(1, (1, 1), activation = 'sigmoid')(u10)
    model = tf.keras.Model(inputs = [inputImage], outputs = [output])
    return model

"""# Evaluation Metric"""

def dice_loss(y_true, y_pred):
  y_true = tf.cast(y_true, tf.float32)
  y_pred = tf.math.sigmoid(y_pred)
  numerator = 2 * tf.reduce_sum(y_true * y_pred)
  denominator = tf.reduce_sum(y_true + y_pred)
  return 1 - numerator / denominator

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def jaccard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)  # -1 ultiplied as we want to minimize this value as loss function

def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)

import keras
METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      jaccard_coef
]

def Plotter(img, predMask, groundTruth):
    plt.figure(figsize=(9,9))  
    plt.subplot(1,4,1)
    plt.imshow(img) #show original img 
    plt.title('image')
    plt.subplot(1,4,2)
    plt.imshow(groundTruth) # show ground truth
    plt.title('actual Mask')
    plt.subplot(1,4,3)
    plt.imshow(predMask) #show hasil prediksi segmentasi
    plt.title('Predicted Mask')
    imh = predMask
    imh[imh < 0.5] = 0
    imh[imh > 0.5] = 1
    plt.subplot(1,4,4)
    plt.imshow(cv2.merge((imh, imh, imh)) * img) # diambil hasil segmentasinya
    plt.title('segmented Image')
    plt.show()

# Import Data
# defining function for dataLoading function
framObjTrain = {'img' : [], 'mask' : [] }

def LoadData( frameObj = None, imgPath = None, maskPath = None, shape = 256):
    imgNames = os.listdir(imgPath)
    maskNames = []
    ## generating mask names
    for mem in tqdm(imgNames):
        maskNames.append(re.sub('\.png', '_seg0.png', mem))
    imgAddr = imgPath + '/'
    maskAddr = maskPath + '/'
    for i in tqdm(range (len(imgNames))):
        try:
            img = plt.imread(imgAddr + imgNames[i])
            mask = plt.imread(maskAddr + maskNames[i]) 
        except:
            continue
        img = cv2.resize(img, (shape, shape))
        mask = cv2.resize(mask, (shape, shape))
        frameObj['img'].append(img)
        frameObj['mask'].append(mask[:,:,0]) # this is because its a binary mask and img is present in channel 0  
    return frameObj


framObjTrain = LoadData(framObjTrain, imgPath = '/content/gdrive/MyDrive/Colab Notebooks/Dataset/Kaggle/leedsbutterfly/actual/img', maskPath = '/content/gdrive/MyDrive/Colab Notebooks/Dataset/Kaggle/leedsbutterfly/mask/img', shape = 256)

## displaying data loaded by our function
# plt.subplot(1,2,1)
# plt.imshow(framObjTrain['img'][1])
# plt.subplot(1,2,2)
# plt.imshow(framObjTrain['mask'][1])
# plt.show()

"""# Training"""

AVERAGE_ACCURACY_TRAIN = []
AVERAGE_JACCARD_TRAIN = []
AVERAGE_RECALL_TRAIN = []
AVERAGE_PRECISION_TRAIN = []

AVERAGE_ACCURACY_TEST = []
AVERAGE_JACCARD_TEST = []
AVERAGE_RECALL_TEST = []
AVERAGE_PRECISION_TEST = []
save_dir = 'saved_models/'

def get_model_name(k):
    return 'model_'+str(k)+'.h5'

#K-CROSS VALIDATION
X = np.array(framObjTrain['img'])
Y = np.array(framObjTrain['mask'])
kf = KFold(n_splits = 10, shuffle=True, random_state=8)                 
i = 1
for train_index, val_index in kf.split(X, Y):
  print("Fold - ", i)
  training_data = X[train_index]
  label_training = Y[train_index]
  testing_data = X[val_index]
  label_testing = Y[val_index]
  
  inputs = tf.keras.layers.Input((256, 256, 3))
  model = GiveMeDEnet(inputs, droupouts= 0.07) # GiveMeUnet ; GiveMeDEnet ; GiveMeUnetWithVGG
  opt = tf.keras.optimizers.Adam(learning_rate=0.01)
  model.compile(optimizer = opt, loss = dice_loss, metrics = METRICS)

  checkpoint = tf.keras.callbacks.ModelCheckpoint(save_dir+get_model_name(i), 
							monitor='val_accuracy', verbose=1, 
							save_best_only=True, mode='max')
	
  callbacks_list = [checkpoint]

  retVal = model.fit(training_data, label_training, epochs = 20, verbose = 1, batch_size=8)

  AVERAGE_ACCURACY_TRAIN.append(np.mean(retVal.history['accuracy']))
  AVERAGE_RECALL_TRAIN.append(np.mean(retVal.history['recall']))
  AVERAGE_JACCARD_TRAIN.append(np.mean(retVal.history['jaccard_coef']))
  AVERAGE_PRECISION_TRAIN.append(np.mean(retVal.history['precision']))

  # plt.plot(retVal.history['loss'], label = 'training_loss')
  # plt.plot(retVal.history['accuracy'], label = 'training_accuracy')
  # plt.legend()
  # plt.grid(True)
  # plt.show()

  i = i + 1

  # evaluate the model
  loss, accuracy, precision, recall, jaccard = model.evaluate(testing_data, label_testing, verbose=0, batch_size=8)

  AVERAGE_ACCURACY_TEST.append(np.mean(accuracy))
  AVERAGE_JACCARD_TEST.append(np.mean(jaccard))
  AVERAGE_RECALL_TEST.append(np.mean(recall))
  AVERAGE_PRECISION_TEST.append(np.mean(precision))

# history = model.fit(training_data, epochs=20, callbacks=callbacks_list, validation_data=label_training, batch_size=8)

print("Accuracy Training : ", np.mean(AVERAGE_ACCURACY_TRAIN))
print("Jaccard Training : ", np.mean(AVERAGE_JACCARD_TRAIN))
print("Recall Training : ", np.mean(AVERAGE_RECALL_TRAIN))
print("Precision Training : ", np.mean(AVERAGE_PRECISION_TRAIN))

print("Accuracy Test : ", np.mean(AVERAGE_ACCURACY_TEST))
print("Jaccard Test : ", np.mean(AVERAGE_JACCARD_TEST))
print("Recall Test : ", np.mean(AVERAGE_RECALL_TEST))
print("Precision Test : ", np.mean(AVERAGE_PRECISION_TEST))


# Count Segmentation time
time_list = []
for i in range(len(testing_data)):
  start = time.time()
  predictions_1 = model.predict(testing_data[i:(i+1)])
  end = time.time()
  duration = (end - start)/60
  time_list.append(duration)
  print(i)
  Plotter(testing_data[i], predictions_1[0][:,:,0], label_testing[i])
print("Rata-rata: ", np.mean(time_list))
# -----------------------------------------------

# Testing 1 citra dan Visualisasi Hasil
predictions_1 = model.predict(testing_data[1:2])
Plotter(testing_data[1], predictions_1[0][:,:,0], label_testing[1])

# tf.keras.utils.plot_model(model, show_shapes=True)

# Save the entire model as a SavedModel.
# !mkdir -p saved_model
# model.save('my_model_DE-Net')

# ------------------ Experiment using another image -----------------------
img_1 = plt.imread('/content/gdrive/MyDrive/Colab Notebooks/Dataset/Kaggle/leedsbutterfly/Picture1.png')
img_1 = cv2.resize(img_1, (256, 256))
# plt.imshow(img_1)
a = cv2.cvtColor(img_1, cv2.COLOR_RGBA2RGB)
list_img_test =  np.array([a])

predictions_1 = model.predict(list_img_test[0:1])
plt.figure(figsize=(15,15))  

plt.subplot(1,2,1)
plt.imshow(predictions_1[0][:,:,0]) #show hasil prediksi segmentasi
plt.title('Predicted Mask')

imh = predictions_1[0][:,:,0]
imh[imh < 0.5] = 0
imh[imh > 0.5] = 1

plt.subplot(1,2,2)
plt.imshow(cv2.merge((imh, imh, imh)) * list_img_test[0]) # diambil hasil segmentasinya
plt.title('segmented Image')
plt.show()
