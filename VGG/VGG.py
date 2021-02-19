import tensorflow as tf
import numpy as np
import pathlib
import datetime
from tf.keras.applications.vgg16 import VGG16
from tf.keras.preprocessing import image
from tf.keras.applications.vgg16 import preprocess_input

# Setting the seeds:
tf.random.set_seed(1)
np.random.seed(1)

VALIDATION_SEED = 0

DATA_PATH = "./train"
METHOD = "adam"
EPOCHS = 100

for i in range(1, len(sys.argv)):
    if (sys.argv[i] == "--folder"):
        DATA_PATH = str(sys.argv[i+1])
    elif (sys.argv[i] == "--method"):
        METHOD = str(sys.argv[i+1])
    elif (sys.argv[i] == "--epochs"):
        EPOCHS = int(sys.argv[i+1])

print(f"DATA_PATH: {DATA_PATH}")
print(f"METHOD: {METHOD}")


# Raw Dataset Directory
data_dir = pathlib.Path(DATA_PATH)

# Shape of inputs to NN Model
BATCH_SIZE = 32
IMG_HEIGHT = 227
IMG_WIDTH = 227

model = VGG16(include_top=True, 
            weights=None,
            input_tensor=None, 
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), 
            pooling=max,
            classes= output_class_units, 
            classifier_activation="softmax")

VAL_SPLIT = 0.2

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), #Resizing the raw dataset
                                                     color_mode='grayscale',
                                                     classes = list(CLASS_NAMES),
                                                     subset = 'training')

val_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH), #Resizing the raw dataset
                                                     color_mode='grayscale',
                                                     classes = list(CLASS_NAMES),
                                                     subset = 'validation')

model.compile(optimizer=METHOD, loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)


# TensorBoard.dev Visuals
log_dir(f"VGG/{METHOD}/logs_{METHOD}_\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(
      train_data_gen,
      steps_per_epoch = np.floor(train_data_gen.samples / BATCH_SIZE),
      epochs = EPOCHS,
      callbacks = [tensorboard_callback, es],
      validation_data = val_data_gen,
      validation_steps = np.floor(val_data_gen.samples / BATCH_SIZE),
      validation_freq = 1)

model.save(f'VGG/{METHOD}/VGG_saved_model/')
