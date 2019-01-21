import gc
import glob
import os
import time

import models_keras
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.callbacks import *


class EpochTimer(keras.callbacks.Callback):

    def __init__(self, filename):
        self.filename = filename
        self.i = 0

    def on_train_begin(self, logs={}):
        self.times = []
        self.start_time = time.time()
        self.file = open('{}'.format(self.filename), 'w')

    def on_epoch_begin(self, epoch, logs={}):
        self.start_time = time.time()

    def on_epoch_end(self, epoch, logs={}):

        epoch_time = time.time() - self.start_time
        self.i += 1
        self.file.write('{},{}\n'.format(self.i, epoch_time))
        self.file.flush()
        self.times.append(epoch_time)

    def on_train_end(self, logs=None):
        self.file.close()


def benchmark_model(X_train, y_train,
                    model_name, run_name,
                    epochs=1, batch_size=128,
                    verbose=False):

    start_time = time.time()

    model = getattr(models_keras, model_name)()
    if verbose:
        print(model.summary())

    compilation_time = time.time() - start_time
    with open('logs/{0}/{0}_compilation_time.log'.format(run_name), 'w') as f:
        f.write('{:.5f}'.format(compilation_time))

    csv_logger = CSVLogger('logs/{0}/{0}.log'.format(run_name))
    epoch_timer = EpochTimer('logs/{0}/{0}_times.log'.format(run_name))

    model.fit(X_train, y_train, callbacks=[csv_logger, epoch_timer],
              epochs=epochs, batch_size=batch_size)

    return
