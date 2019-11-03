import os
import tensorflow as tf
import numpy as np
import cv2
import keras
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, Conv2DTranspose, Dense, Flatten, concatenate
from keras.optimizers import *
from keras import backend as kb
from keras.losses import mse
from keras.callbacks import ModelCheckpoint

def psnr(y_true,y_pred):
	return 10*kb.log(kb.mean(kb.square(y_true-y_pred)))/kb.log(10.)

def psnr1(y_true,y_pred):
	return 10*np.log(np.mean(np.square(y_true-y_pred)))/np.log(10.)

class Derain(object):
    def __init__(self, data_dir='./training/', checkpoint_dir='./checkpoints/'):
        #Change
        self.num_files = 2000
        self.batch_size=8
        #self.epochs = 1
        self.epochs=50
        self.num_feature = 128 #512 in the paper
        #self.num_feature = 32
        self.learning_rate = 1e-3
        # input image dimensions
        self.img_x, self.img_y = 512,512 #256,256
        self.label_size = 512 #256
        self.image_size = 512 #256
        self.num_channels = 3
        #------------FILE PATH------------#
        self.filepath = "."
        self.filepath_weights = "."

    def train(self, training_steps=10):
        x_train = []
        y_train = []

        pixels = 0.0
        for i in range(1,701):
            img = cv2.imread("training/"+str(i)+".jpg")
            img = img/255.0
            #print(img.shape)
            #base = low_filter(img, img.shape[0], img.shape[1])
            #detail = img - base

            [res1, res2] = np.hsplit(img.astype('float'), 2) #[res1, res2] = np.hsplit(detail.astype('float'), 2)
            pixels += res1.shape[0] * res1.shape[1]
            res1 -= res2


            result1 = np.zeros(shape=(512,512,3))
            res1 = res1[:512, :512]
            result1[:res1.shape[0],:res1.shape[1],:res1.shape[2]] = res1

            result2 = np.zeros(shape=(512,512,3))
            res2 = res2[:512, :512]
            result2[:res2.shape[0],:res2.shape[1],:res2.shape[2]] = res2


            x_train.append(result2)
            y_train.append(result1)


        '''
        model = Sequential()
        #First convolutional layer
        model.add(Conv2D(self.num_feature, kernel_size=(16, 16), strides=(1, 1),padding='SAME',
            activation='tanh',kernel_initializer='random_normal',bias_initializer='constant',
            input_shape=(self.label_size,self.label_size,self.num_channels)))
        #Second con layer
        model.add(Conv2D(self.num_feature, kernel_size=(1,1), strides=(1, 1),padding='SAME',activation='tanh',
            kernel_initializer='random_normal',bias_initializer='constant'))
        #Third convolutional layer
        model.add(Conv2DTranspose(self.num_channels,kernel_size=(8,8),strides=(1,1),padding='SAME',
            activation='linear',kernel_initializer='random_normal',bias_initializer='constant',
            output_shape=(self.label_size,self.label_size,self.num_channels)))
        model.compile(loss=custom_objective,#keras.losses.mean_squared_error,
            optimizer=keras.optimizers.Adam(),metrics = ['accuracy'])
        history = AccuracyHistory()
        '''


        inputs = Input((512, 512, 3))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(3, (3, 3), activation='relu', padding='same')(conv9)


        model = Model(input = inputs, output = conv9)
        checkpoint = ModelCheckpoint('saved_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = [checkpoint]

        model.compile(optimizer = Adam(lr = 1e-4), loss =psnr)

        model.fit(np.array(x_train), np.array(y_train),
            batch_size=self.batch_size,
            epochs=self.epochs,
            verbose=1,
            validation_split = 0.1,
            callbacks=callbacks_list
        )

        self.save_model(model)


    def save_model(self, model, steps=10):
        model.save('model.h5')
        model.save_weights('model_weights.h5')
        print("Saved model to disk")


    def load_model(self):
        model = keras.models.load_model('saved.hdf5')
        model.load_weights('saved.hdf5')
        print("Loaded model from disk")
        return model


    def low_filter(data, height, width):
        r = 15
        eps = 1.0
        channel = 3
        batch_q = np.zeros((height, width, channel))
        for j in range(channel):
                I = data[:, :, j]
                p = data[:, :, j]
                ones_array = np.ones([height, width])
                N = cv2.boxFilter(ones_array, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0)
                mean_I = cv2.boxFilter(I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                mean_p = cv2.boxFilter(p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                mean_Ip = cv2.boxFilter(I * p, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                cov_Ip = mean_Ip - mean_I * mean_p
                mean_II = cv2.boxFilter(I * I, -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                var_I = mean_II - mean_I * mean_I
                a = cov_Ip / (var_I + eps)
                b = mean_p - a * mean_I
                mean_a = cv2.boxFilter(a , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                mean_b = cv2.boxFilter(b , -1, (2 * r + 1, 2 * r + 1), normalize = False, borderType = 0) / N
                q = mean_a * I + mean_b
                batch_q[:, :,j] = q
        return batch_q


dr = Derain(".")
#dr.train()

model = keras.models.load_model('saved.hdf5', custom_objects  = {"psnr":psnr})
model.load_weights('saved.hdf5')
print("Loaded model from disk")

x_test = []
y_test = []

for i in range(1,98):
	img = cv2.imread("./test/"+str(i)+".jpg")
	img = img/255.0
	#print(img.shape)
	#base = low_filter(img, img.shape[0], img.shape[1])
	#detail = img - base

	[res1, res2] = np.hsplit(img.astype('float'), 2) #[res1, res2] = np.hsplit(detail.astype('float'), 2)
	res1 -= res2

	result1 = np.zeros(shape=(512,512,3))
	res1 = res1[:512, :512]
	result1[:res1.shape[0],:res1.shape[1],:res1.shape[2]] = res1

	result2 = np.zeros(shape=(512,512,3))
	res2 = res2[:512, :512]
	result2[:res2.shape[0],:res2.shape[1],:res2.shape[2]] = res2

	x_test.append(result2)
	y_test.append(result1)

prediction = model.predict(np.array(x_test), verbose=1)

print(psnr1(np.array(x_test), np.array(y_test)))
print(mse(prediction, y_test))
