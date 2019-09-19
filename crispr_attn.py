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
from torch import save, load
from time import time
import random
import my_data
from torch import cuda, device
import shap
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

def run():

    data_pre = OT_crispr_attn.data_preparer()
    print(data_pre.get_crispr_preview())
    X = data_pre.prepare_x()
    y = data_pre.get_labels()
    data_pre.persist_data()
    print(X.head())
    print(data_pre.feature_length_map)
    train_index, test_index = data_pre.train_test_split()
    if config.log2fc_filter:
        train_filter_1 = (y.loc[train_index, :] > 10)
        y[train_filter_1] = 10
        train_filter_2 = (y.loc[train_index, :] < -10)
        y[train_filter_2] = -10
        # train_filter = (y.loc[train_index, :] > 10)
        # train_index = train_index[train_filter.iloc[:,0]]
    print(train_index, test_index)
    torch.manual_seed(0)
    # X = X.values.astype(np.float32)
    # y = y.values.astype(np.float32)
    # y_binary = (y > np.quantile(y, 0.8)).astype(int).reshape(-1, )

    #y[y.loc[train_index, config.y] < 0] = 0
    # essen_filter = list(data_pre.crispr[data_pre.crispr['log2fc'] > 0].index)
    # train_index = list(set(train_index).intersection(essen_filter))
    # test_index = list(set(test_index).intersection(essen_filter))
    std_scaler = StandardScaler()
    m_m = MinMaxScaler((0,100))
    if config.y_transform:
        std_scaler.fit(y.loc[train_index, :])
        new_y = std_scaler.transform(y) * 100
        y = pd.DataFrame(new_y, columns=y.columns, index = y.index)
        m_m.fit(y.loc[train_index, :])
        new_y = m_m.transform(y)
        y = pd.DataFrame(new_y, columns=y.columns, index = y.index)


    if config.test_cellline is not None:
        test_cellline_index = data_pre.crispr[data_pre.crispr['cellline'].isin(config.test_cellline)].index
        test_index = test_cellline_index.intersection(test_index)

    if config.train_cellline is not None:
        train_cellline_index = data_pre.crispr[data_pre.crispr['cellline'].isin(config.train_cellline)].index
        train_index = train_cellline_index.intersection(train_index)

    logger.debug("training data amounts: %s, testing data amounts: %s" % (len(train_index), len(test_index)))
    x_train, x_test, y_train, y_test = \
        X.loc[train_index, :], X.loc[test_index, :], \
        y.loc[train_index, :], y.loc[test_index, :]

    _, unique_train_index = np.unique(pd.concat([x_train, y_train], axis=1), return_index=True, axis=0)
    _, unique_test_index = np.unique(pd.concat([x_test, y_test], axis=1), return_index=True, axis=0)
    logger.debug("after deduplication, training data amounts: %s, testing data amounts: %s" % (
    len(unique_train_index), len(unique_test_index)))
    train_index = train_index[unique_train_index]
    test_index = test_index[unique_test_index]
    x_concat = pd.concat([X, y], axis=1)
    _, unique_index = np.unique(x_concat, return_index=True, axis=0)
    logger.debug("{0!r}, {1!r}".format((len(x_concat.loc[train_index, :].drop_duplicates())), str(len(x_concat.loc[train_index, :]))))
    logger.debug("Splitted dataset successfully")

    train_eval_index_list = list(train_index)
    random.shuffle(train_eval_index_list)
    sep = int(len(train_eval_index_list) * 0.9)
    train_index_list = train_eval_index_list[:sep]
    eval_index_list = train_eval_index_list[sep:]
    partition = {'train': train_eval_index_list,
                 'eval': list(test_index),
                 'test': list(test_index)}

    labels = {key: value for key, value in zip(range(len(y)),
                                               list(y.values.reshape(-1)))}

    train_params = {'batch_size': config.batch_size,
                    'shuffle': True}
    eval_params = {'batch_size': len(test_index),
                   'shuffle': False}
    test_params = {'batch_size': len(test_index),
                   'shuffle': False}

    logger.debug("Preparing datasets ... ")
    training_set = my_data.MyDataset(partition['train'], labels)
    training_generator = data.DataLoader(training_set, **train_params)

    train_bg_params = {'batch_size': len(train_eval_index_list)//6 + 5,
                    'shuffle': False}
    training_bg_set = my_data.MyDataset(partition['train'], labels)
    training_bg_generator = data.DataLoader(training_bg_set, **train_bg_params)

    validation_set = my_data.MyDataset(partition['eval'], labels)
    validation_generator = data.DataLoader(validation_set, **eval_params)

    test_set = my_data.MyDataset(partition['test'], labels)
    test_generator = data.DataLoader(test_set, **test_params)

    logger.debug("I might need to augment data")

    logger.debug("Building the scaled dot product attention model")
    for_input_len = data_pre.feature_length_map[0][1] - data_pre.feature_length_map[0][0]
    extra_input_len = 0 if not data_pre.feature_length_map[2] \
        else data_pre.feature_length_map[2][1] - data_pre.feature_length_map[2][0]

    crispr_model = attention_model.get_OT_model(data_pre.feature_length_map)
    best_crispr_model = attention_model.get_OT_model(data_pre.feature_length_map)
    crispr_model.to(device2)
    best_crispr_model.to(device2)
    # crispr_model = attention_model.get_model(d_input=for_input_len)
    # best_crispr_model = attention_model.get_model(d_input=for_input_len)
    # #crispr_model = attention_model.get_OT_model(data_pre.feature_length_map, classifier=True)
    # crispr_model.to(device2)
    # best_crispr_model.to(device2)
    best_cv_spearman_score = 0

    if config.retraining:

        logger.debug("I need to load a old trained model")
        crispr_model = load(config.retraining_model)
        crispr_model.load_state_dict(load(config.retraining_model_state))
        crispr_model.to(device2)
        logger.debug("I might need to freeze some of the weights")

    logger.debug("Built the RNN model successfully")

    optimizer = torch.optim.Adam(crispr_model.parameters(), lr=config.start_lr, weight_decay=config.lr_decay,
                                 betas=(0.9, 0.98), eps=1e-9)

    try:
        if config.training:

            logger.debug("Training the model")
            mse_visualizer = torch_visual.VisTorch(env_name='MSE')
            pearson_visualizer = torch_visual.VisTorch(env_name='Pearson')
            spearman_visualizer = torch_visual.VisTorch(env_name='Spearman')

            for epoch in range(config.n_epochs):

                start = time()
                cur_epoch_train_loss = []
                train_total_loss = 0
                i = 0

                # Training
                for local_batch, local_labels in training_generator:
                    i += 1
                    # Transfer to GPU
                    local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
                    # seq_local_batch = local_batch.narrow(dim=1, start=0, length=for_input_len).long()
                    # extra_local_batch = local_batch.narrow(dim=1, start=for_input_len, length=extra_input_len)
                    # Model computations
                    preds = crispr_model(local_batch).contiguous().view(-1)
                    ys = local_labels.contiguous().view(-1)
                    optimizer.zero_grad()
                    assert preds.size(-1) == ys.size(-1)
                    #loss = F.nll_loss(preds, ys)
                    crispr_model.train()
                    loss = F.mse_loss(preds, ys)
                    loss.backward()
                    optimizer.step()

                    train_total_loss += loss.item()

                    n_iter = 2
                    if i % n_iter == 0:
                        sample_size = len(train_index)
                        p = int(100 * i * config.batch_size / sample_size)
                        avg_loss = train_total_loss / n_iter
                        if config.y_inverse_transform:
                            avg_loss = \
                            std_scaler.inverse_transform(np.array(avg_loss / 100).reshape(-1, 1)).reshape(-1)[0]
                        logger.debug("   %dm: epoch %d [%s%s]  %d%%  loss = %.3f" % \
                                                 ((time() - start) // 60, epoch, "".join('#' * (p // 5)),
                                                  "".join(' ' * (20 - (p // 5))), p, avg_loss))
                        train_total_loss = 0
                        cur_epoch_train_loss.append(avg_loss)

                ### Evaluation
                val_i = 0
                val_total_loss = 0
                val_loss = []
                val_preds = []
                val_ys = []
                val_pearson = 0
                val_spearman = 0

                with torch.set_grad_enabled(False):

                    crispr_model.eval()
                    for local_batch, local_labels in training_bg_generator:
                        val_i += 1
                        local_labels_on_cpu = np.array(local_labels).reshape(-1)
                        sample_size = local_labels_on_cpu.shape[-1]
                        local_labels_on_cpu = local_labels_on_cpu[:sample_size]
                        # Transfer to GPU
                        local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
                        # seq_local_batch = local_batch.narrow(dim=1, start=0, length=for_input_len).long()
                        # extra_local_batch = local_batch.narrow(dim=1, start=for_input_len, length=extra_input_len)
                        preds = crispr_model(local_batch).contiguous().view(-1)
                        assert preds.size(-1) == local_labels.size(-1)
                        prediction_on_cpu = preds.cpu().numpy().reshape(-1)
                        # mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                        #                                   prediction_on_cpu[sample_size:]], axis=0)
                        mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
                        if config.y_inverse_transform:
                            local_labels_on_cpu, mean_prediction_on_cpu = \
                                std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1, 1) / 100), \
                                std_scaler.inverse_transform(mean_prediction_on_cpu.reshape(-1, 1) / 100)
                        loss = mean_squared_error(local_labels_on_cpu, mean_prediction_on_cpu)
                        val_preds.append(mean_prediction_on_cpu)
                        val_ys.append(local_labels_on_cpu)
                        val_total_loss += loss

                        n_iter = 1
                        if val_i % n_iter == 0:
                            avg_loss = val_total_loss / n_iter
                            val_loss.append(avg_loss)
                            val_total_loss = 0

                    mean_prediction_on_cpu = np.concatenate(tuple(val_preds))
                    local_labels_on_cpu = np.concatenate(tuple(val_ys))
                    val_pearson = pearsonr(mean_prediction_on_cpu.reshape(-1), local_labels_on_cpu.reshape(-1))[0]
                    val_spearman = spearmanr(mean_prediction_on_cpu.reshape(-1), local_labels_on_cpu.reshape(-1))[0]

                logger.debug(
                    "Validation mse is {0}, Validation pearson correlation is {1!r} and Validation "
                    "spearman correlation is {2!r}".format(np.mean(val_loss),val_pearson, val_spearman))
                mse_visualizer.plot_loss(epoch, np.mean(cur_epoch_train_loss), np.mean(val_loss), loss_type='mse')
                pearson_visualizer.plot_loss(epoch, val_pearson, loss_type='pearson_loss', ytickmin=0, ytickmax=1)
                spearman_visualizer.plot_loss(epoch, val_spearman, loss_type='spearman_loss', ytickmin=0, ytickmax=1)

                ### Evaluation
                val_i = 0
                val_total_loss = 0
                val_loss = []
                val_preds = []
                val_ys = []
                val_pearson = 0
                val_spearman = 0

                with torch.set_grad_enabled(False):

                    crispr_model.eval()
                    for local_batch, local_labels in validation_generator:
                        val_i += 1
                        local_labels_on_cpu = np.array(local_labels).reshape(-1)
                        sample_size = local_labels_on_cpu.shape[-1]
                        local_labels_on_cpu = local_labels_on_cpu[:sample_size]
                        # Transfer to GPU
                        local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
                        # seq_local_batch = local_batch.narrow(dim=1, start=0, length=for_input_len).long()
                        # extra_local_batch = local_batch.narrow(dim=1, start=for_input_len, length=extra_input_len)
                        preds = crispr_model(local_batch).contiguous().view(-1)
                        assert preds.size(-1) == local_labels.size(-1)
                        prediction_on_cpu = preds.cpu().numpy().reshape(-1)
                        # mean_prediction_on_cpu = np.mean([prediction_on_cpu[:sample_size],
                        #                                   prediction_on_cpu[sample_size:]], axis=0)
                        mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
                        if config.y_inverse_transform:
                            local_labels_on_cpu, mean_prediction_on_cpu = \
                                std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1, 1) / 100), \
                                std_scaler.inverse_transform(mean_prediction_on_cpu.reshape(-1, 1) / 100)
                        loss = mean_squared_error(local_labels_on_cpu, mean_prediction_on_cpu)
                        val_preds.append(mean_prediction_on_cpu)
                        val_ys.append(local_labels_on_cpu)
                        val_total_loss += loss

                        n_iter = 1
                        if val_i % n_iter == 0:
                            avg_loss = val_total_loss / n_iter
                            val_loss.append(avg_loss)
                            val_total_loss = 0

                    mean_prediction_on_cpu = np.concatenate(tuple(val_preds))
                    local_labels_on_cpu = np.concatenate(tuple(val_ys))
                    val_pearson = pearsonr(mean_prediction_on_cpu.reshape(-1), local_labels_on_cpu.reshape(-1))[0]
                    val_spearman = spearmanr(mean_prediction_on_cpu.reshape(-1), local_labels_on_cpu.reshape(-1))[0]

                    if best_cv_spearman_score < val_spearman:
                        best_cv_spearman_score = val_spearman
                        best_crispr_model.load_state_dict(crispr_model.state_dict())

                logger.debug(
                    "Test mse is {0}, Test pearson correlation is {1!r} and Test "
                    "spearman correlation is {2!r}".format(np.mean(val_loss),val_pearson, val_spearman))
                mse_visualizer.plot_loss(epoch, np.mean(cur_epoch_train_loss), np.mean(val_loss), loss_type='mse')
                pearson_visualizer.plot_loss(epoch, val_pearson, loss_type='pearson_loss', ytickmin=0, ytickmax=1)
                spearman_visualizer.plot_loss(epoch, val_spearman, loss_type='spearman_loss', ytickmin=0, ytickmax=1)

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

    logger.debug("Persisting model")
    # serialize weights to HDF5
    save(best_crispr_model, config.hdf5_path)
    save(best_crispr_model.state_dict(), config.hdf5_path_state)
    logger.debug("Saved model to disk")

    ### Testing
    test_i = 0
    test_total_loss = 0
    test_loss = []
    test_pearson = 0
    test_preds = []
    test_ys = []

    with torch.set_grad_enabled(False):

        best_crispr_model.eval()
        for local_batch, local_labels in test_generator:
            # Transfer to GPU
            test_i += 1
            local_labels_on_cpu = np.array(local_labels).reshape(-1)
            sample_size = local_labels_on_cpu.shape[-1]
            local_labels_on_cpu = local_labels_on_cpu[:sample_size]
            local_batch, local_labels = local_batch.float().to(device2), local_labels.float().to(device2)
            # seq_local_batch = local_batch.narrow(dim=1, start=0, length=for_input_len).long()
            # extra_local_batch = local_batch.narrow(dim=1, start=for_input_len, length=extra_input_len)
            # Model computations
            preds = best_crispr_model(local_batch).contiguous().view(-1)
            assert preds.size(-1) == local_labels.size(-1)
            prediction_on_cpu = preds.cpu().numpy().reshape(-1)
            mean_prediction_on_cpu = prediction_on_cpu[:sample_size]
            if config.y_inverse_transform:
                local_labels_on_cpu, mean_prediction_on_cpu = \
                    std_scaler.inverse_transform(local_labels_on_cpu.reshape(-1, 1) / 100), \
                    std_scaler.inverse_transform(prediction_on_cpu.reshape(-1, 1) / 100)
            loss = mean_squared_error(local_labels_on_cpu, mean_prediction_on_cpu)
            test_total_loss += loss
            test_preds.append(mean_prediction_on_cpu)
            test_ys.append(local_labels_on_cpu)

            n_iter = 1
            if (test_i) % n_iter == 0:
                avg_loss = test_total_loss / n_iter
                test_loss.append(avg_loss)
                test_total_loss = 0

        mean_prediction_on_cpu = np.concatenate(tuple(test_preds))
        local_labels_on_cpu = np.concatenate(tuple(test_ys))
        test_pearson = pearsonr(local_labels_on_cpu.reshape(-1), mean_prediction_on_cpu.reshape(-1))[0]
        test_spearman = spearmanr(local_labels_on_cpu.reshape(-1), mean_prediction_on_cpu.reshape(-1))[0]
        save_output = []
        if 'essentiality' in config.extra_numerical_features:
            save_output.append(data_pre.crispr.loc[test_index, 'essentiality'])
        save_output.append(pd.DataFrame(local_labels_on_cpu, columns=['ground_truth']))
        save_output.append(pd.DataFrame(mean_prediction_on_cpu, columns=['prediction']))
        save_output = pd.concat(save_output, ignore_index=True, axis=1)
        save_output.to_csv(config.test_prediction, index = False)




    logger.debug("Testing mse is {0}, Testing pearson correlation is {1!r} and Testing "
                 "spearman correlation is {2!r}".format(np.mean(test_loss), test_pearson, test_spearman))


    if config.check_feature_importance:

        logger.debug("Getting features ranks")
        names = []
        names += ["src_" + str(i) for i in range(data_pre.feature_length_map[0][1])]
        if data_pre.feature_length_map[1] is not None: names += ["trg_" + str(i)
                                                                 for i in range(data_pre.feature_length_map[1][1] - data_pre.feature_length_map[1][0])]
        if data_pre.feature_length_map[2] is not None: names += config.extra_categorical_features + config.extra_numerical_features
        ranker = feature_imp.InputPerturbationRank(names)
        feature_ranks = ranker.rank(2, y.loc[train_eval_index_list, :].values, best_crispr_model, [torch.FloatTensor(X.loc[train_eval_index_list, :].values)], torch=True)
        feature_ranks_df = pd.DataFrame(feature_ranks)
        feature_ranks_df.to_csv(config.feature_importance_path, index = False)
        logger.debug("Get features ranks successfully")

    # batch_input_importance = []
    # batch_out_input_importance = []
    # total_data, _ = next(iter(training_bg_generator))
    # total_data = total_data.float().to(device2)
    # seq_total_data = total_data.narrow(dim=1, start=0, length=for_input_len).long()
    # extra_total_data = total_data.narrow(dim=1, start=for_input_len, length=extra_input_len)
    # for local_batch, local_labels in test_generator:
    #     # Transfer to GPU
    #     local_batch = local_batch.float().to(device2)
    #     seq_local_batch = local_batch.narrow(dim=1, start=0, length=for_input_len).long()
    #     extra_local_batch = local_batch.narrow(dim=1, start=for_input_len, length=extra_input_len)
    #     if config.save_feature_imp_model:
    #         save(best_drug_model, config.hdf5_path)
    #     # Model computations
    #     e = shap.GradientExplainer(best_drug_model, data=list(total_data))
    #     input_importance = e.shap_values(list(local_batch))
    #     #pickle.dump(input_shap_values, open(setting.input_importance_path, 'wb+'))
    #     batch_input_importance.append(input_importance)
    #     logger.debug("Finished one batch of input importance analysis")
    #
    #     e1 = shap.GradientExplainer((best_drug_model, best_drug_model.out), data=list(total_data))
    #     out_input_shap_value = e1.shap_values(list(local_batch))
    #     batch_out_input_importance.append(out_input_shap_value)
    #     logger.debug("Finished one batch of out input importance analysis")
    #
    #     if setting.save_inter_imp:
    #         transform_input_importance = []
    #         for layer in best_drug_model.dropouts:
    #             cur_e = shap.GradientExplainer((best_drug_model, layer), data=list(total_data))
    #             cur_transform_input_shap_value = cur_e.shap_values(list(local_batch))
    #             transform_input_importance.append(cur_transform_input_shap_value)
    #         transform_input_importance = np.concatenate(tuple(transform_input_importance), axis=1)
    #
    #         batch_transform_input_importance.append(transform_input_importance)
    #     logger.debug("Finished one batch of importance analysis")
    # batch_input_importance = np.concatenate(tuple(batch_input_importance), axis=0)
    # batch_out_input_importance = np.concatenate(tuple(batch_out_input_importance), axis=0)
    # pickle.dump(batch_input_importance, open(setting.input_importance_path, 'wb+'))
    # pickle.dump(batch_out_input_importance, open(setting.out_input_importance_path, 'wb+'))
    # if setting.save_inter_imp:
    #     batch_transform_input_importance = np.concatenate(tuple(batch_transform_input_importance), axis=0)
    #     pickle.dump(batch_transform_input_importance, open(setting.transform_input_importance_path, 'wb+'))
    # logger.debug("Closing sessions")
    # mse_visualizer.close()
    # pearson_visualizer.close()
    # spearman_visualizer.close()

    #
    # logger.debug("Saving test and prediction data plot")
    # if last_performance > performance:
    #     prediction = last_prediction
    # utils.ytest_and_prediction_output(y_test[unique_test_index], prediction)
    # logger.debug("Saved test and prediction data plot successfully")
    #
    # if config.check_feature_importance:
    #     if performance > last_performance:
    #         loaded_model = load_model(config.temp_hdf5_path,
    #                                   custom_objects={'revised_mse_loss': utils.revised_mse_loss, 'tf': tf})
    #         crispr_model = models.CrisprCasModel.compile_transfer_learning_model(loaded_model)
    #     logger.debug("Getting features ranks")
    #     names = []
    #     names += ["for_" + str(i) for i in range(for_input.shape[1])]
    #     names += ["rev_" + str(i) for i in range(rev_input.shape[1])]
    #     names += ["off_" + str(i) for i in range(off_target_X_train.shape[1])]
    #     names += config.extra_categorical_features + config.extra_numerical_features
    #     ranker = feature_imp.InputPerturbationRank(names)
    #     feature_ranks = ranker.rank(20, y_test[unique_test_index], crispr_model,
    #                                 [data[unique_test_index] for data in test_list])
    #     feature_ranks_df = pd.DataFrame(feature_ranks)
    #     feature_ranks_df.to_csv(config.feature_importance_path, index=False)
    #     logger.debug("Get features ranks successfully")

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
