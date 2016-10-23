from __future__ import print_function

import cv2
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D, Deconvolution2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K

from data import load_train_data, load_test_data

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code

img_rows = 512
img_cols = 512

smooth = 1.


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def get_fcn32():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='linear', border_mode='same', input_shape=(1, 512, 512))(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Convolution2D(64, 3, 3, activation='linear', border_mode='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Convolution2D(128, 3, 3, activation='linear', border_mode='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Convolution2D(256, 3, 3, activation='linear', border_mode='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Convolution2D(512, 3, 3, activation='linear', border_mode='same')(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    deconv = Deconvolution2D(1, 32, 32, activation='sigmoid', output_shape=(4, 1, 512, 512), subsample=(32,32)) (pool5)
    model = Model(input=inputs, output=deconv)   
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    print (model.output_shape)
    return model

def get_fcn16():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='linear', border_mode='same', input_shape=(1, 512, 512))(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Convolution2D(64, 3, 3, activation='linear', border_mode='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Convolution2D(128, 3, 3, activation='linear', border_mode='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Convolution2D(256, 3, 3, activation='linear', border_mode='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Convolution2D(512, 3, 3, activation='linear', border_mode='same')(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    upscore = UpSampling2D(size=(2, 2))(pool5)
    merged = merge([upscore, conv5], mode='sum')
    deconv = Deconvolution2D(1, 16, 16, activation='sigmoid', output_shape=(4, 1, 512, 512), subsample=(16,16)) (merged)
      
    model = Model(input=inputs, output=deconv)   
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    print (model.output_shape)
    return model

def get_fcn8():
    inputs = Input((1, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='linear', border_mode='same', input_shape=(1, 512, 512))(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Convolution2D(64, 3, 3, activation='linear', border_mode='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Convolution2D(128, 3, 3, activation='linear', border_mode='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Convolution2D(128, 3, 3, activation='linear', border_mode='same')(pool3)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    conv5 = Convolution2D(128, 3, 3, activation='linear', border_mode='same')(pool4)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    
    upscore = UpSampling2D(size=(2,2))(pool4)
    upscore1 = UpSampling2D(size=(4, 4))(pool5)
    merged1 = merge([upscore, upscore1, pool3], mode='sum')
    deconv = Deconvolution2D(1, 8, 8, activation='sigmoid', output_shape=(4, 1, 512, 512), subsample=(8,8)) (merged1)
      
    model = Model(input=inputs, output=deconv)   
    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    print (model.output_shape)
    return model

def preprocess(imgs):
    imgs_p = np.ndarray((imgs.shape[0], imgs.shape[1], img_rows, img_cols), dtype=np.uint8)
    for i in range(imgs.shape[0]):
        imgs_p[i, 0] = cv2.resize(imgs[i, 0], (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
    return imgs_p


def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    imgs_train, imgs_mask_train = load_train_data()

    imgs_train = preprocess(imgs_train)
    imgs_mask_train = preprocess(imgs_mask_train)

    imgs_train = imgs_train.astype('float32')
    #mean = np.mean(imgs_train)  # mean for data centering
    #std = np.std(imgs_train)  # std for data normalization

    #imgs_train -= mean
    #imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

    print('Creating and compiling model...')
    # model = get_fcn32()
    # model = get_fcn16()
    model = get_fcn8()
    model_checkpoint = ModelCheckpoint('fcn32.hdf5', monitor='loss', save_best_only=True)

    print('Fitting model...')
    model.fit(imgs_train, imgs_mask_train, batch_size=4, nb_epoch=100, verbose=1, shuffle=True, callbacks=[model_checkpoint])

    print('Loading and preprocessing test data...')
    imgs_test, imgs_id_test = load_test_data()
    imgs_test = preprocess(imgs_test)

    imgs_test = imgs_test.astype('float32')

    print('Loading saved weights...')
    model.load_weights('fcn32.hdf5')

    print('Predicting masks on test data...')
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test)


if __name__ == '__main__':
    train_and_predict()
