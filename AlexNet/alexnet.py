import tensorflow as tf
import numpy as np
import pathlib
import datetime
import os
import sys



VALIDATION_SEED = 0

DATA_PATH = "./train"
TEST_PATH = "./test"
METHOD = "adam"
EPOCHS = 20
SEED = 1

for i in range(1, len(sys.argv)):
    if (sys.argv[i] == "--folder"):
        DATA_PATH = str(sys.argv[i+1])
    elif (sys.argv[i] == "--method"):
        METHOD = str(sys.argv[i+1])
    elif (sys.argv[i] == "--epochs"):
        EPOCHS = int(sys.argv[i+1])
    elif (sys.argv[i] == "--seed"):
    	SEED = int(sys.argv[i+1])
    elif (sys.argv[i] == "--test"):
        TEST_PATH = str(sys.argv[i+1])

# Setting the seeds:
tf.random.set_seed(SEED)
np.random.seed(SEED)

print(f"DATA_PATH: {DATA_PATH}")
print(f"METHOD: {METHOD}")





# Raw Dataset Directory
data_dir = pathlib.Path(DATA_PATH)

image_count = len(list(data_dir.glob('*/*.png')))
CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
CLASS_NAMES.sort()
output_class_units = len(CLASS_NAMES)

print(list(CLASS_NAMES))

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(output_class_units, activation='softmax')
])

# Shape of inputs to NN Model
BATCH_SIZE = 32
IMG_HEIGHT = 227
IMG_WIDTH = 227

VAL_SPLIT = 0.2

image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)

train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     color_mode='grayscale',
                                                      #Resizing the raw dataset
                                                     classes = list(CLASS_NAMES),
                                                     subset = 'training')

val_data_gen = image_generator.flow_from_directory(directory=str(data_dir),
                                                     batch_size=BATCH_SIZE,
                                                     shuffle=True,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     color_mode='grayscale',
                                                      #Resizing the raw dataset
                                                     classes = list(CLASS_NAMES),
                                                     subset = 'validation')

test_image_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_data_dir = pathlib.Path(TEST_PATH)

print("Test images:")

test_gen = test_image_gen.flow_from_directory(str(test_data_dir),
                                                target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                classes = list(CLASS_NAMES),
                                                color_mode='grayscale',
                                                batch_size=1)

model.compile(optimizer=METHOD, loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

#es = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=1, min_delta=0.01)


# TensorBoard.dev Visuals
log_dir=f"Alex_{SEED}/{METHOD}/logs_{METHOD}_\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

#callback_weights = tf.keras.callbacks.ModelCheckpoint(f"./{METHOD}/checkpoint" + "_{epoch}", verbose=1, save_weights_only=False, save_freq='epoch')

es = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience=10, restore_best_weights=True)

# Training the model
history = model.fit(
      train_data_gen,
      steps_per_epoch = np.floor(train_data_gen.samples / BATCH_SIZE),
      epochs = EPOCHS,
      callbacks = [tensorboard_callback, es],
      validation_data = val_data_gen,
      validation_steps = np.floor(val_data_gen.samples / BATCH_SIZE),
      validation_freq = 1)


model.save(f'Alex_{SEED}/{METHOD}/AlexNet_saved_model/')

print("performing eval...")

eval = model.evaluate(test_gen)

print("results:")
print(eval)
