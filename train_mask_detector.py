#Import library for model training 
import tensorflow as tf
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.python import keras
from tensorflow import lite
print(tf.__version__)
tf.test.gpu_device_name()

#Data images processing
img_generator = ImageDataGenerator(rescale = 1 / 255.0,
                                   zoom_range = 0.1,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   shear_range = 0.2,
                                   horizontal_flip = True,
                                   fill_mode = 'nearest')

train = img_generator.flow_from_directory('/home/chafik/Internship/dataset_mask', 
                                          target_size = (224, 224),
                                          classes = ['with_mask','without_mask'],
                                          class_mode = 'categorical', 
                                          batch_size = 64, 
                                          shuffle = True)

#Model design
based_model = MobileNetV2(weights = 'imagenet',
include_top = False,
input_shape = (224, 224, 3))
based_model.trainable = False

model = Sequential()
model.add(based_model)
model.add(GlobalAveragePooling2D())
model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5)) #ป้องกัน Model overfitting
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(2))
model.add(Activation('softmax'))
model.summary()
print(model)
# Model optimizer
opt = Adam(lr = 0.001, decay = 0.001 / 20)
model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])


#Model traning
model.fit(train, batch_size = 64, epochs = 15)

#Save model
karas_file = "face_mask.h5"
keras.models.save_model(model,karas_file)

saved_model_path = "/home/chafik/Internship/mask_unmask.h5" 
model.save(saved_model_path)
saved_model_path = tf.keras.models.load_model('mask_unmask.h5')
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path) 
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)


