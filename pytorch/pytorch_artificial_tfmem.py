import gc
import glob
import logging
import os

import numpy as np
import pandas as pd
import pretrainedmodels
import torch
import torchvision
from models_pytorch import *
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torch_dataset import BenchmarkDataset
from torchvision import models
from utils_pytorch import *

FIT_MAX_BATCH = True

NUM_SAMPLES = 10000
SIZE = 299
NUM_CHANNELS = 3
EPOCHS = 5
NUM_RUNS = 3

all_models = [
    # 'DenseNet121',
    # 'DenseNet169',
    # 'Inception3',
    'InceptionResNetV2',
    # 'NASNet',
    # 'PNASNet',
    'ResNet50'
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
        if MODEL_NAME == 'DenseNet121':
            MAX_BATCH_SIZE = 12
        if MODEL_NAME == 'DenseNet169':
            MAX_BATCH_SIZE = 8
        if MODEL_NAME == 'Inception3':
            MAX_BATCH_SIZE = 32
        if MODEL_NAME == 'InceptionResNetV2':
            MAX_BATCH_SIZE = 12  # was 16
        if MODEL_NAME == 'NASNet':
            MAX_BATCH_SIZE = 4
        if MODEL_NAME == 'PNASNet':
            MAX_BATCH_SIZE = 4
        if MODEL_NAME == 'ResNet50':
            MAX_BATCH_SIZE = 16  # was 24

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

    train_dataset = BenchmarkDataset(
        X_train, y_train)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        pin_memory=True,
        shuffle=True)

    loss = torch.nn.BCELoss().cuda(0)

    for i in range(NUM_RUNS):

        run_name = '{}_size{}_batch{}_trial_{}'.format(
            MODEL_NAME, SIZE, BATCH_SIZE, i)
        print('Running: {}\n'.format(run_name))

        if not os.path.isdir('./logs/{}'.format(run_name)):
            os.makedirs('./logs/{}'.format(run_name))

        pd.DataFrame.from_dict(parameters_dict, orient='index').to_csv(
            './logs/{0}/{0}_parameters.csv'.format(run_name), header=None)

        benchmark_model(train_loader,
                        loss,
                        model_parameters,
                        MODEL_NAME, run_name,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE)
