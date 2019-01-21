import gc
import glob
import os
import time

import models_pytorch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm


def benchmark_model(train_loader,
                    loss_fn,
                    model_parameters,
                    model_name, run_name,
                    epochs=1, batch_size=128,
                    gpu=0,
                    verbose=False):

    start_time = time.time()
    epoch_times = {}

    model = getattr(models_pytorch, model_name)(model_parameters)
    if verbose:
        print(model)

    model.train()
    model.cuda(gpu)

    compilation_time = time.time() - start_time
    with open('logs/{0}/{0}_compilation_time.log'.format(run_name), 'w') as f:
        f.write('{:.5f}'.format(compilation_time))

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log_file = open('logs/{0}/{0}.log'.format(run_name), 'w')

    for e in range(epochs):

        epoch_start = time.time()

        for image, target in tqdm(train_loader):

            image = image.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)
            y_pred = model(image)
            y_pred = F.sigmoid(y_pred)

            loss = loss_fn(y_pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - epoch_start
        # epoch_times[e] =
        log_file.write('{},{}\n'.format(e, epoch_time))
        log_file.flush()

    log_file.close()

    return
