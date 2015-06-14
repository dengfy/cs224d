#!/bin/bash

# verbose
set -x


infile="models/RNN2_wvecDim_30_middleDim_95_step_1e-2_2.bin" # the pickled neural network
model="RNN2" # the neural network type

echo $infile

# test the model on test data
#python runNNet.py --inFile $infile --test --data "test" --model $model

# test the model on dev data
python runNNet.py --inFile $infile --test --data "dev" --model $model

# test the model on training data
#python runNNet.py --inFile $infile --test --data "train" --model $model












