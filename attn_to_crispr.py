#!/usr/bin/env python
import logging
import os
import importlib
import sys
import numpy as np
import pandas as pd
import attention_model
import torch
from torch.utils import data
import torch_visual
from torch import save, load
from time import time
import random
import my_data
from torch import cuda, device
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import OT_crispr_attn
import feature_imp

# setting nvidia gpu environment
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# CUDA for PyTorch
use_cuda = cuda.is_available()
if use_cuda:
    device2 = device("cuda:0")
    cuda.set_device(device2)
    cuda.empty_cache()
else:
    device2 = device("cpu")

torch.set_default_tensor_type('torch.FloatTensor')

# Setting the correct config file
config_path = ".".join(["models", sys.argv[1]]) + "." if len(sys.argv) >= 2 else ""
config = importlib.import_module(config_path + "config")

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(config.run_specific_log, mode='a')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Recurrent neural network")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

def run():

    data_pre = OT_crispr_attn.data_preparer()
    print(data_pre.get_crispr_preview())
    X = data_pre.prepare_x()
    y = data_pre.get_labels()
    data_pre.persist_data()
    print(X.head())
    print(data_pre.feature_length_map)
    torch.manual_seed(0)
    partition = {'test': list(X.index)}
    labels = {key: value for key, value in zip(range(len(y)),
                                               list(y.values.reshape(-1)))}
    test_params = {'batch_size': len(config.batch_size),
                   'shuffle': False}
    logger.debug("Preparing datasets ... ")
    test_set = my_data.MyDataset(partition['test'], labels)
    test_generator = data.DataLoader(test_set, **test_params)
    logger.debug("Building the scaled dot product attention model")

    logger.debug("loading a trained model")
    crispr_model = load(config.retraining_model)
    crispr_model.load_state_dict(load(config.retraining_model_state))
    crispr_model.to(device2)

    ### Testing
    test_i = 0
    test_total_loss = 0
    test_loss = []
    test_pearson = 0
    test_preds = []
    test_ys = []

    with torch.set_grad_enabled(False):

        crispr_model.eval()
        for local_batch, local_labels in test_generator:
            # Transfer to GPU
            test_i += 1
            local_labels_on_cpu = np.array(local_labels).reshape(-1)
            sample_size = local_labels_on_cpu.shape[-1]
            local_labels_on_cpu = local_labels_on_cpu[:sample_size]
            local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
            preds = crispr_model(local_batch).contiguous().view(-1)
            assert preds.size(-1) == local_labels.size(-1)
            prediction_on_cpu = preds.cpu().numpy().reshape(-1)
            mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
            loss = mean_squared_error(local_labels_on_cpu, mean_prediction_on_cpu)
            test_total_loss += loss
            test_preds.append(mean_prediction_on_cpu)
            test_ys.append(local_labels_on_cpu)

            n_iter = 50
            if (test_i) % n_iter == 0:
                avg_loss = test_total_loss / n_iter
                test_loss.append(avg_loss)
                test_total_loss = 0

        mean_prediction_on_cpu = np.concatenate(tuple(test_preds))
        local_labels_on_cpu = np.concatenate(tuple(test_ys))
        test_pearson = pearsonr(local_labels_on_cpu.reshape(-1), mean_prediction_on_cpu.reshape(-1))[0]
        test_spearman = spearmanr(local_labels_on_cpu.reshape(-1), mean_prediction_on_cpu.reshape(-1))[0]
        save_output = []
        save_output.append(pd.DataFrame(local_labels_on_cpu, columns=['ground_truth']))
        save_output.append(pd.DataFrame(mean_prediction_on_cpu, columns=['prediction']))
        save_output = pd.concat(save_output, ignore_index=True, axis=1)
        save_output.to_csv(config.test_prediction, index = False)

    logger.debug("Testing mse is {0}, Testing pearson correlation is {1!r} and Testing "
                 "spearman correlation is {2!r}".format(np.mean(test_loss), test_pearson, test_spearman))
