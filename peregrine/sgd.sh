#!/bin/bash
#SBATCH --time=03:30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --output=sgd.out

# Get the resources we need
module load TensorFlow/2.3.1-fosscuda-2019b-Python-3.7.4
pip install pycuda --user
pip install pillow --user


# copy the dataset to the temporary folder

cp $HOME/DL/train.tar.xz /local/tmp/
tar xf /local/tmp/train.tar.xz -C /local/tmp/

cp $HOME/DL/test.tar.xz /local/tmp/
tar xf /local/tmp/test.tar.xz -C /local/tmp/

# Create the new datapoints
python datasetAugmenter.py --folder /local/tmp/train

#Run the training
python alexnet.py --folder /local/tmp/train --epochs 100 --method sgd --seed 1 --test /local/tmp/test
