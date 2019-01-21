import gc
import glob
import os
import time

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from utils_keras import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


FIT_MAX_BATCH = True

NUM_SAMPLES = 10000
SIZE = 299
NUM_CHANNELS = 3
EPOCHS = 5
NUM_RUNS = 3


all_models = [
    # 'densenet121',
    # 'densenet169',
    # 'inceptionv3',
    # 'inceptionresnetv2',
    'resnet50',
    'nasnet_large',
]


input_dim = (NUM_SAMPLES, SIZE, SIZE, NUM_CHANNELS)
print('\ninput dim: {}'.format(input_dim))

X_train = np.random.randint(
    0, 255, input_dim, dtype=np.uint8).astype(np.float32)
y_train = np.expand_dims((np.random.rand(NUM_SAMPLES) > 0.5).astype(
    np.uint8), axis=-1).astype(np.float32)

print('X: {}'.format(X_train.shape))
print('y: {}'.format(y_train.shape))


model_parameters = {
    'num_classes': 1,
    'pretrained': False,
    'num_channels': 3,
    'pooling_output_dim': 1,
    'model_name': '',
}


for m in all_models:

    print('running: {}'.format(m))

    MODEL_NAME = m

    if FIT_MAX_BATCH:
        if MODEL_NAME == 'densenet121':
            MAX_BATCH_SIZE = 12  # 16 gives memory warning
        if MODEL_NAME == 'densenet169':
            MAX_BATCH_SIZE = 8  # 12 gives memory warning
        if MODEL_NAME == 'inceptionv3':
            MAX_BATCH_SIZE = 32  # 32 works
        if MODEL_NAME == 'inceptionresnetv2':
            MAX_BATCH_SIZE = 12
        if MODEL_NAME == 'nasnet_large':
            MAX_BATCH_SIZE = 4  # doesn't work ??, not 6
        if MODEL_NAME == 'resnet50':
            MAX_BATCH_SIZE = 16

    parameters_dict = {
        'MODEL_NAME': MODEL_NAME,
        'NUM_SAMPLES': NUM_SAMPLES,
        'NUM_CHANNELS': NUM_CHANNELS,
        'SIZE': SIZE,
        'EPOCHS': EPOCHS,
        'BATCH_SIZE': MAX_BATCH_SIZE,
        'NUM_RUNS': NUM_RUNS,
    }
    print('parameters:\n{}'.format(parameters_dict))

    if FIT_MAX_BATCH:
        parameters_dict['BATCH_SIZE'] = MAX_BATCH_SIZE
        print('Fit max batch size into memory.')

    BATCH_SIZE = parameters_dict['BATCH_SIZE']

    for i in range(NUM_RUNS):

        run_name = '{}_size{}_batch{}_trial_{}'.format(
            MODEL_NAME, SIZE, BATCH_SIZE, i)
        print('Running: {}\n'.format(run_name))

        if not os.path.isdir('./logs/{}'.format(run_name)):
            os.makedirs('./logs/{}'.format(run_name))

        pd.DataFrame.from_dict(parameters_dict, orient='index').to_csv(
            './logs/{0}/{0}_parameters.csv'.format(run_name), header=None)

        benchmark_model(X_train, y_train,
                        MODEL_NAME, run_name,
                        epochs=EPOCHS, batch_size=BATCH_SIZE)
