import tensorflow as tf
import numpy as np
import pathlib
import os

def eval(netFolder, testData):

    # Set up testing data
    IMG_HEIGHT = 227
    IMG_WIDTH = 227

    data_dir = pathlib.Path(testData)
    CLASS_NAMES = np.array([item.name for item in data_dir.glob('*')])
    CLASS_NAMES.sort()
    image_count = len(list(data_dir.glob('*/*.png')))
    output_class_units = len(CLASS_NAMES)

    image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    test_it = image_generator.flow_from_directory(  str(data_dir),
                                                    target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                    classes = list(CLASS_NAMES),
                                                    color_mode='grayscale',
                                                    batch_size=1)

    # Find and evaluate all models:
    results = {}
    for f in os.scandir(netFolder):
        if f.is_dir():
            algName = f.name

            modelFolder = None

            #Find the model folder
            for f2 in os.scandir(f"{netFolder}/{f.name}"):
                if "saved_model" in f2.name:
                    modelFolder = f2.name

            if (modelFolder == None):
                print(f"Could not find model for folder: {algName}")
                return

            modelPath = f"{netFolder}/{f.name}/{modelFolder}/"

            model = tf.keras.models.load_model(modelPath)

            modelResults = model.evaluate(test_it)

            results[algName] = modelResults
    return results
