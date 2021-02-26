import numpy as np
from PIL import Image, ImageOps
import os
import sys

#Note: All datapoints we generated are prefixed with gen for easy
#identification

def create_flipped(folder):

    print(f"Creating flipped images for: {folder}")

    ctr = 0

    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            im = Image.open(f"{folder}/{filename}")
            imMirrored = ImageOps.mirror(im)
            imMirrored.save(f"{folder}/gen_mirrored_{ctr}.png")

            ctr += 1

def add_noise(image):
    pixels = image.load()

    for row in range(image.height):
        for col in range(image.width):
            noise = int(np.random.normal(0, 0.01) * 255)
            pixels[col, row] = pixels[col, row] + noise


def generate_noisy_vers(folder, amount):

    print("Generating noisy images")

    ctr = 0

    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            im = Image.open(f"{folder}/{filename}")

            for i in range(amount):
                add_noise(im)
                im.save(f"{folder}/gen_noise_{ctr}_var_{i}.png")

            ctr += 1


def changeBrightness(folder, amount):

        print("Changing brightness values")
        ctr = 0

        for filename in os.listdir(folder):
            if filename.endswith(".png"):
                im = Image.open(f"{folder}/{filename}")

                pixels = im.load()

                for row in range(im.height):
                    for col in range(im.width):
                        pixels[col, row] = pixels[col, row] + amount

                im.save(f"{folder}/gen_bright_shift_{ctr}.png")

                ctr += 1

def cleanup(folder):
    print(f"Cleaning up: {folder}")

    for filename in os.listdir(folder):
        if filename.startswith("gen") and filename.endswith(".png"):
            os.remove(f"{folder}/{filename}")




target_folder = "train"
perform_cleanup = False

if (len(sys.argv) != 1):

    for i in range(1, len(sys.argv)):

        if (sys.argv[i] == "--folder"):
            target_folder = sys.argv[i+1]
        elif (sys.argv[i] == "--cleanup"):
            perform_cleanup = True

if (perform_cleanup):
    folders = ["angry", "disgusted", "fearful", "neutral", "sad", "surprised"]

    for folder in folders:
        cleanup(f"{target_folder}/{folder}")

else:
    create_flipped(f"{target_folder}/angry")
    create_flipped(f"{target_folder}/disgusted")
    create_flipped(f"{target_folder}/fearful")
    create_flipped(f"{target_folder}/neutral")
    create_flipped(f"{target_folder}/sad")
    create_flipped(f"{target_folder}/surprised")

    changeBrightness(f"{target_folder}/disgusted", 10)
    changeBrightness(f"{target_folder}/disgusted", -15) # Note that this will not revert the above, but generate more images

    generate_noisy_vers(f"{target_folder}/disgusted", 3)
