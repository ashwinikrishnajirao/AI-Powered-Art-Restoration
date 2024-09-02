from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import cv2, os
from PIL import Image

# Define paths
chkptPath = './weights/weights.hdf5'
test_path2 = './one'
save_path = './test_resized'
save_path2 = './test_output'

# Define a simpler model architecture
input_img = Input(shape=(256, 384, 3))
x = Conv2D(16, (3, 3), strides=2, activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((1, 2), padding='same')(x)

x = Flatten()(x)
x = Dropout(0.4)(x)
x = Dense(64 * 4 * 3, activation='relu')(x)

x = Reshape((4, 3, 64))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((1, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

model = Model(input_img, x)
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Load weights
model.load_weights(chkptPath)

# Predict function
for i in os.listdir(test_path2 + '/lol'):
    # Predict
    img = cv2.imread(test_path2 + '/lol/' + i)
    img = cv2.resize(img, (192, 128))  # Adjust image size
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    res = model.predict(img)
    
    for a in res:
        a *= 255.0
        im = Image.fromarray(a.astype('uint8'))
        im.save(save_path2 + '/out_' + i)
    
    # Save original image
    img = cv2.imread(test_path2 + '/lol/' + i)
    img = np.array(img)  # / 255.0
    img = cv2.resize(img, (192, 128))  # Adjust image size
    im = Image.fromarray(img.astype('uint8'))
    im.save(save_path + '/' + i)
