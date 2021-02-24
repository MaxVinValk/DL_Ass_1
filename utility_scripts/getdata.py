import os
import tensorflow as tf
import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator

def nameAndValFromString(string):

    name = str(string[string.find(":") + 3:string.find("\n") - 1]).strip()
    val = float(string[string.find("simple_value") + 14: -1].strip())


    return name, val



def fileToData(filepath):

    valuesFound = {}
    maxEpoch = 0

    for s in summary_iterator(filepath):
        if (s.step > maxEpoch):
            maxEpoch = s.step

    for s in summary_iterator(filepath):

        step = s.step
        data = str(s.summary.value)

        if "epoch_loss" not in data and "epoch_accuracy" not in data:
            continue

        name, val = nameAndValFromString(str(s.summary.value))

        if name not in valuesFound.keys():
            valuesFound[name] = [0] * (maxEpoch + 1) # Epochs 0-index

        valuesFound[name][step] = val

    return valuesFound

def getV2File(folder):
    for f in os.scandir(folder):
        if not f.is_dir() and f.name.endswith(".v2"):
            return f
    print("No V2 file found...")
    exit(1)

def getData(folder):

    results = {}

    for f in os.scandir(folder):
        if f.is_dir():
            # We get the name
            name = f.name[5:]
            name = name[:name.find("_")]

            results[name] = {}

            trainDataFile = getV2File(f"{folder}/{f.name}/train").name
            validationDataFile = getV2File(f"{folder}/{f.name}/validation").name

            results[name]["train"] = fileToData(f"{folder}/{f.name}/train/{trainDataFile}")
            results[name]["validation"] = fileToData(f"{folder}/{f.name}/validation/{validationDataFile}")

    return results







'''
def readLogs(logfolder):
    pass

def findAllFiles(rootDir, maxSeedIdx = 5):

    results = {}
    results["alex"] = {}
    results["vgg"] = {}

    for f in os.scandir(rootDir):
        if f.is_dir() and "seed" in f.name:

            seedStr = f.name[f.name.find("seed_") + 5:]
            seedStrEnd = seedStr.find("_")

            seed = int(f.name[seedStr: seedStr + seedStrEnd])

            for f2 in os.scandir(f"{rootDir}/{f.name}"):
                if f2.is_dir() and "logs" in f2.name:
                    data = readLogs(f"{rootDir}/{f.name}/{f2.name}")
                    name = f2.name[5 : f2.name.find("_", 5)]

                    if "vgg" in f.name:
                        if name not in results["vgg"].keys():
                            results["vgg"][name] = [0] * (maxSeedIdx + 1)
                        results["vgg"][name][seed] = data
                    else:
                        if name not in results["alex"].keys():
                            results["alex"][name] = [0] * (maxSeedIdx + 1)
                        results["alex"][name][seed] = data
'''
