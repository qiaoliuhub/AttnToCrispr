#!/usr/bin/env python

import logging
import os
import importlib
import sys
import pickle
import numpy as np
import pandas as pd
import utils
import process_features
import attention_model
import attention_setting
import torch
from torch.utils import data
import torch_visual
from torch import save
from time import time
import random
import my_data
from torch import cuda, device
import feature_imp
import shap
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import OT_crispr_attn
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import roc_auc_score, average_precision_score

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
config_path = ".".join(sys.argv[1].split("/")[-3:]) + "." if len(sys.argv) >= 2 and sys.argv[1].split("/")[
    -1].startswith("run") else ""
config = importlib.import_module(config_path + "config")

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(config.run_specific_log, mode='a')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Recurrent neural network")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)


# output will be the average spearman correlation
def get_prediction_group(model, index_list, unique_list, output_data, test_data):
    total_per = 0
    prediction = []

    num_of_unuse_index = 0
    for test_index in index_list:
        new_index = [i for i in test_index if i in unique_list]
        cur_prediction = model.predict(x=[data[new_index, :] for data in test_data])
        cur_per = spearmanr(cur_prediction, output_data[new_index, :])[0]
        prediction.extend(list(cur_prediction))
        if len(new_index) <= 2:
            num_of_unuse_index += 1
            continue
        total_per += cur_per

    print(num_of_unuse_index)
    return total_per / float(len(index_list) - num_of_unuse_index), pd.Series(prediction)


def get_prediction_regular(model, test_index, unique_list, output_data, test_data):
    prediction = model.predict(x=[data[unique_list, :] for data in test_data])
    # prediction = prediction.round(2)
    performance = spearmanr(prediction, output_data[unique_list, :])[0]

    return performance, pd.Series(list(prediction))

def run():

    data_pre = OT_crispr_attn.data_preparer()
    print(data_pre.get_crispr_preview())
    X = data_pre.prepare_x(mismatch=True, trg_seq_col='Target sequence')
    y = data_pre.get_labels(binary=True)
    data_pre.persist_data()
    print(X.head())
    logger.debug("{0!r}".format(data_pre.feature_length_map))
    train_index, test_index = data_pre.train_test_split(n_split=5)
    logger.debug("{0!r}".format(set(data_pre.crispr.loc[test_index, :]['sequence'])))
    logger.debug("{0!r}".format(set(data_pre.crispr.loc[train_index, :]['sequence'])))
    #assert len(set(data_pre.crispr.loc[test_index, :]['sequence']) & set(data_pre.crispr.loc[train_index, :]['sequence'])) == 0
    logger.debug("{0!r}".format(train_index))
    logger.debug("{0!r}".format(test_index))
    logger.debug("training data amounts: %s, testing data amounts: %s" % (len(train_index), len(test_index)))
    torch.manual_seed(0)

    if config.test_cellline:
        test_cellline_index = data_pre.crispr[data_pre.crispr['cellline'] == config.test_cellline].index
        test_index = test_cellline_index.intersection(test_index)

    ros = RandomOverSampler(random_state=42)
    _ = ros.fit_resample(X.loc[train_index,:], y.loc[train_index,:])
    new_train_index = train_index[ros.sample_indices_]
    oversample_train_index = list(new_train_index)
    random.shuffle(oversample_train_index)

    # sep = int(len(train_eval_index_list) * 0.9)
    # train_index_list = train_eval_index_list[:sep]
    # eval_index_list = train_eval_index_list[sep:]

    assert len(set(oversample_train_index) & set(test_index)) == 0
    assert len(set(oversample_train_index) & set(train_index)) == len(set(train_index))
    partition = {'train': oversample_train_index,
                 'train_val': train_index,
                 'eval': list(test_index),
                 'test': list(test_index)}

    labels = {key: value for key, value in zip(list(range(len(y))),
                                               list(y.values.reshape(-1)))}

    train_params = {'batch_size': config.batch_size,
                    'shuffle': True}
    train_bg_params = {'batch_size': config.batch_size,
                    'shuffle': True}
    eval_params = {'batch_size': len(test_index),
                   'shuffle': False}
    test_params = {'batch_size': len(test_index),
                   'shuffle': False}

    logger.debug("Preparing datasets ... ")
    training_set = my_data.MyDataset(partition['train'], labels)
    training_generator = data.DataLoader(training_set, **train_params)

    training_bg_set = my_data.MyDataset(partition['train_val'], labels)
    training_bg_generator = data.DataLoader(training_bg_set, **train_bg_params)

    validation_set = my_data.MyDataset(partition['eval'], labels)
    validation_generator = data.DataLoader(validation_set, **eval_params)

    test_set = my_data.MyDataset(partition['test'], labels)
    test_generator = data.DataLoader(test_set, **test_params)

    logger.debug("I might need to augment data")

    logger.debug("Building the scaled dot product attention model")
    crispr_model = attention_model.get_OT_model(data_pre.feature_length_map, classifier=True)
    best_crispr_model = attention_model.get_OT_model(data_pre.feature_length_map, classifier=True)
    crispr_model.to(device2)
    best_crispr_model.to(device2)
    best_cv_roc_auc_scores = 0
    optimizer = torch.optim.Adam(crispr_model.parameters(), lr=config.start_lr, weight_decay=config.lr_decay,
                                 betas=(0.9, 0.98), eps=1e-9)

    if config.retraining:

        logger.debug("I need to load a old trained model")
        logger.debug("I might need to freeze some of the weights")

    logger.debug("Built the RNN model successfully")

    try:
        if config.training:

            logger.debug("Training the model")
            nllloss_visualizer = torch_visual.VisTorch(env_name='NLLLOSS')
            roc_auc_visualizer = torch_visual.VisTorch(env_name='ROC_AUC')
            pr_auc_visualizer = torch_visual.VisTorch(env_name='PR_AUC')

            for epoch in range(config.n_epochs):

                crispr_model.train()
                start = time()
                cur_epoch_train_loss = []
                train_total_loss = 0
                i = 0

                # Training
                for local_batch, local_labels in training_generator:
                    i += 1
                    # Transfer to GPU
                    local_batch, local_labels = local_batch.float().to(device2), local_labels.long().to(device2)
                    # seq_local_batch = local_batch.narrow(dim=1, start=0, length=for_input_len).long()
                    # extra_local_batch = local_batch.narrow(dim=1, start=for_input_len, length=extra_input_len)
                    # Model computations
                    preds = crispr_model(local_batch)
                    ys = local_labels.contiguous().view(-1)
                    optimizer.zero_grad()
                    assert preds.size(0) == ys.size(0)
                    loss = F.nll_loss(preds, ys)
                    loss.backward()
                    optimizer.step()

                    train_total_loss += loss.item()

                    n_iter = 2
                    if i % n_iter == 0:
                        sample_size = len(train_index)
                        p = int(100 * i * config.batch_size / sample_size)
                        avg_loss = train_total_loss / n_iter
                        logger.debug("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                                                 ((time() - start) // 60, epoch, "".join('#' * (p // 5)),
                                                  "".join(' ' * (20 - (p // 5))), p, avg_loss))
                        train_total_loss = 0
                        cur_epoch_train_loss.append(avg_loss)

                ### Evaluation
                train_val_i = 0
                train_val_total_loss = 0
                train_val_loss = []
                # train_val_roc_auc = 0
                # train_val_pr_auc = 0
                train_val_preds = []
                train_val_ys = []
                n_pos, n_neg = 0, 0

                with torch.set_grad_enabled(False):

                    crispr_model.eval()
                    for local_batch, local_labels in training_bg_generator:
                        train_val_i += 1
                        local_labels_on_cpu = np.array(local_labels).reshape(-1)
                        train_val_ys.append(local_labels_on_cpu)
                        n_pos += sum(local_labels_on_cpu)
                        n_neg += len(local_labels_on_cpu) - sum(local_labels_on_cpu)
                        # Transfer to GPU
                        local_batch, local_labels = local_batch.float().to(device2), local_labels.long().to(device2)
                        preds = crispr_model(local_batch)
                        assert preds.size(0) == local_labels.size(0)
                        nllloss_val = F.nll_loss(preds, local_labels).item()
                        train_val_total_loss += nllloss_val
                        prediction_on_cpu = preds.cpu().numpy()[:,1]
                        train_val_preds.append(prediction_on_cpu)

                        n_iter = 10
                        if train_val_i % n_iter == 0:
                            avg_loss = train_val_total_loss / n_iter
                            train_val_loss.append(avg_loss)
                            train_val_total_loss = 0

                preds = np.concatenate(tuple(train_val_preds))
                ys = np.concatenate(tuple(train_val_ys))
                train_val_roc_auc = roc_auc_score(ys, preds)
                train_val_pr_auc = average_precision_score(ys, preds)
                logger.debug("{0!r} positive samples and {1!r} negative samples".format(n_pos, n_neg))
                logger.debug(
                    "Validation nllloss is {0}, Validation roc_auc is {1!r} and Validation "
                    "pr_auc correlation is {2!r}".format(np.mean(train_val_loss), train_val_roc_auc, train_val_pr_auc))
                # nllloss_visualizer.plot_loss(epoch, np.mean(cur_epoch_train_loss), np.mean(train_val_loss), loss_type='nllloss')
                # roc_auc_visualizer.plot_loss(epoch, train_val_roc_auc, loss_type='roc_auc', ytickmin=0, ytickmax=1)
                # pr_auc_visualizer.plot_loss(epoch, train_val_pr_auc, loss_type='pr_auc', ytickmin=0, ytickmax=1)

                ### Evaluation
                val_i = 0
                val_total_loss = 0
                val_loss = []
                val_preds = []
                val_ys = []

                with torch.set_grad_enabled(False):

                    crispr_model.eval()
                    for local_batch, local_labels in validation_generator:
                        val_i += 1
                        local_labels_on_cpu = np.array(local_labels).reshape(-1)
                        val_ys.append(local_labels_on_cpu)
                        n_pos = sum(local_labels_on_cpu)
                        n_neg = len(local_labels_on_cpu) - sum(local_labels_on_cpu)
                        logger.debug("{0!r} positive samples and {1!r} negative samples".format(n_pos, n_neg))
                        local_batch, local_labels = local_batch.float().to(device2), local_labels.long().to(device2)
                        preds = crispr_model(local_batch)
                        assert preds.size(0) == local_labels.size(0)
                        nllloss_val = F.nll_loss(preds, local_labels).item()
                        prediction_on_cpu = preds.cpu().numpy()[:,1]
                        val_preds.append(prediction_on_cpu)
                        val_total_loss += nllloss_val

                        n_iter = 1
                        if val_i % n_iter == 0:
                            avg_loss = val_total_loss / n_iter
                            val_loss.append(avg_loss)
                            val_total_loss = 0

                preds = np.concatenate(tuple(val_preds))
                ys = np.concatenate(tuple(val_ys))
                val_roc_auc = roc_auc_score(ys, preds)
                val_pr_auc = average_precision_score(ys, preds)
                logger.debug(
                    "Test NLLloss is {0}, Test roc_auc is {1!r} and Test "
                    "pr_auc is {2!r}".format(np.mean(val_loss),val_roc_auc, val_pr_auc))
                nllloss_visualizer.plot_loss(epoch, np.mean(cur_epoch_train_loss), np.mean(val_loss), loss_type='nllloss')
                roc_auc_visualizer.plot_loss(epoch, train_val_roc_auc, val_roc_auc, loss_type='roc_auc', ytickmin=0, ytickmax=1)
                pr_auc_visualizer.plot_loss(epoch, train_val_pr_auc, val_pr_auc, loss_type='pr_auc', ytickmin=0, ytickmax=1)

                if best_cv_roc_auc_scores < val_roc_auc:
                    best_cv_roc_auc_scores = val_roc_auc
                    best_crispr_model.load_state_dict(crispr_model.state_dict())

            logger.debug("Saving training history")

            logger.debug("Saved training history successfully")

            logger.debug("Trained crispr model successfully")

        else:
            logger.debug("loading in old model")

            logger.debug("Load in model successfully")

    except KeyboardInterrupt as e:

        logger.debug("Loading model")
        logger.debug("loading some intermediate step's model")
        logger.debug("Load in model successfully")

    # logger.debug("Persisting model")
    # # serialize weights to HDF5
    # crispr_model.save(config.hdf5_path)
    # logger.debug("Saved model to disk")

    #best_index = np.argmax(cv_roc_auc_scores)
    #best_drug_model = cv_models[int(best_index)]
    ### Testing
    test_i = 0
    test_total_loss = 0
    test_loss = []
    test_roc_auc = 0
    test_pr_auc = 0
    test_preds = []
    test_ys = []

    with torch.set_grad_enabled(False):

        best_crispr_model.eval()
        for local_batch, local_labels in test_generator:
            # Transfer to GPU
            test_i += 1
            local_labels_on_cpu = np.array(local_labels).reshape(-1)
            test_ys.append(local_labels_on_cpu)
            n_pos = sum(local_labels_on_cpu)
            n_neg = len(local_labels_on_cpu) - sum(local_labels_on_cpu)
            logger.debug("{0!r} positive samples and {1!r} negative samples".format(n_pos, n_neg))
            local_batch, local_labels = local_batch.float().to(device2), local_labels.long().to(device2)
            # Model computations
            preds = best_crispr_model(local_batch)
            assert preds.size(0) == local_labels.size(0)
            nllloss_test = F.nll_loss(preds, local_labels).item()
            prediction_on_cpu = preds.cpu().numpy()[:, 1]
            test_preds.append(prediction_on_cpu)
            test_total_loss += nllloss_test

            n_iter = 1
            if (test_i + 1) % n_iter == 0:
                avg_loss = test_total_loss / n_iter
                test_loss.append(avg_loss)
                test_total_loss = 0

        preds = np.concatenate(tuple(test_preds))
        ys = np.concatenate(tuple(test_ys))
        logger.debug("{0!r} test data was tested".format(len(ys)))
        test_roc_auc = roc_auc_score(ys, preds)
        test_pr_auc = average_precision_score(ys, preds)
        save_output = []
        if 'essentiality' in config.extra_numerical_features:
            save_output.append(data_pre.crispr.loc[test_index, 'essentiality'])
        save_output.append(pd.DataFrame(ys, columns=['ground_truth']))
        save_output.append(pd.DataFrame(preds, columns=['prediction']))
        save_output = pd.concat(save_output, ignore_index=True, axis=1)
        save_output.to_csv(config.test_prediction, index=False)
        logger.debug("Testing NLLloss is {0}, Testing roc_auc is {1!r} and Testing "
                     "pr_auc is {2!r}".format(np.mean(test_loss), test_roc_auc, test_pr_auc))



    if config.check_feature_importance:

        logger.debug("Getting features ranks")
        names = []
        names += ["src_" + str(i) for i in range(data_pre.feature_length_map[0][1])]
        if data_pre.feature_length_map[1] is not None: names += ["trg_" + str(i)
                                                                 for i in range(data_pre.feature_length_map[1][1] - data_pre.feature_length_map[1][0])]
        if data_pre.feature_length_map[2] is not None: names += config.extra_categorical_features + config.extra_numerical_features
        ranker = feature_imp.InputPerturbationRank(names)
        feature_ranks = ranker.rank(2, y.loc[train_index, :].values, best_crispr_model, [torch.FloatTensor(X.loc[train_index, :].values)],
                                    torch=True, classifier=True)
        feature_ranks_df = pd.DataFrame(feature_ranks)
        feature_ranks_df.to_csv(config.feature_importance_path, index = False)
        logger.debug("Get features ranks successfully")


if __name__ == "__main__":

    np.random.seed(3)

    try:
        run()
        logger.debug("new directory %s" % config.run_dir)

    except:

        import shutil

        shutil.rmtree(config.run_dir)
        logger.debug("clean directory %s" % config.run_dir)
        raise
