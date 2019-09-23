#!/usr/bin/env python

import logging
import os
import importlib
import sys
import pickle
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.models import load_model
from scipy.stats import spearmanr
from keras.layers import Input
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import h2o
import feature_imp
import utils
import process_features
import models

# setting nvidia gpu environment
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

# Setting the correct config file
config_path = ".".join(["models", sys.argv[1]]) + "." if len(sys.argv) >= 2 else ""
config = importlib.import_module(config_path+"config")
attention_setting = importlib.import_module(config_path+"attention_setting")

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
        cur_prediction = model.predict(x = [data[new_index, :] for data in test_data])
        cur_per = spearmanr(cur_prediction, output_data[new_index, :])[0]
        prediction.extend(list(cur_prediction))
        if len(new_index) <= 2:
            num_of_unuse_index += 1
            continue
        total_per += cur_per

    print(num_of_unuse_index)
    return total_per / float(len(index_list)-num_of_unuse_index), pd.Series(prediction)

def get_prediction_regular(model, test_index, unique_list, output_data, test_data):

    prediction = model.predict(x=[data[unique_list, :] for data in test_data])
    #prediction = prediction.round(2)
    performance = spearmanr(prediction, output_data[unique_list, :])[0]

    return performance, pd.Series(list(prediction))


def ml_train(X, extra_crispr_df, y, train_index, test_index):
    logger.debug("Creating h2o working env")
    # ### Start H2O
    # Start up a 1-node H2O cloud on your local machine, and allow it to use all CPU cores and up to 2GB of memory:
    h2o.init(max_mem_size="6G")
    h2o.remove_all()
    logger.debug("Created h2o working env successfully")

    from h2o.estimators import H2ORandomForestEstimator
    
    rf_crispr = H2ORandomForestEstimator(
        model_id="rf_crispr",
        categorical_encoding="enum",
        nfolds=5,
        ntrees=300,
        # max_depth = 20,
        # nbins = 20,
        stopping_rounds=30,
        score_each_iteration=True,
        seed=100000)
    '''
    rf_crispr = H2OXGBoostEstimator(
        model_id="rf_crispr",
        categorical_encoding="enum",
        nfolds=5,
        ntrees=300,
        stopping_rounds=30,
        score_each_iteration=True,
        seed=1000000)
    '''
    seq_data = X.iloc[:, :config.seq_len]
    seq_data.columns = ['pos_' + str(i) for i in range(len(seq_data.columns))]
    pre_h2o_df = pd.concat([seq_data, extra_crispr_df, y], axis=1)
    h2o_crispr_df_train = h2o.H2OFrame(pre_h2o_df.loc[train_index, :])
    h2o_crispr_df_test = h2o.H2OFrame(pre_h2o_df.loc[test_index, :])

    logger.debug("Training machine learning model")
    rf_crispr.train(x=h2o_crispr_df_train.col_names[:-1], y=h2o_crispr_df_train.col_names[-1],
                    training_frame=h2o_crispr_df_train)
    logger.debug("Trained successfully. Output feature importance")
    feature_importance = rf_crispr._model_json['output']['variable_importances'].as_data_frame()[
        ['variable', 'percentage']]
    feature_importance.to_csv(config.feature_importance_path, index=False)

    logger.debug("Predicting training data")
    test_prediction_train = rf_crispr.predict(h2o_crispr_df_train[:-1])
    performance = spearmanr(test_prediction_train.as_data_frame()['predict'], h2o_crispr_df_train.as_data_frame()['log2fc'])[0]
    logger.debug("spearman correlation coefficient for training dataset is: %f" % performance)

    logger.debug("Predicting test data")
    test_prediction = rf_crispr.predict(h2o_crispr_df_test[:-1])
    performance = spearmanr(test_prediction.as_data_frame()['predict'], h2o_crispr_df_test.as_data_frame()['log2fc'])[0]
    logger.debug("spearman correlation coefficient for training dataset is: %f" % performance)

    logger.debug("Saving model")
    h2o.save_model(rf_crispr, config.ml_model_path)
    logger.debug("Saved model to disk")

def run():

    logger.debug("Reading in the crispr dataset %s" % config.input_dataset)
    crispr = pd.read_csv(config.input_dataset)
    crispr['PAM'] = crispr['sequence'].str[-3:]
    if config.log_cen:
        crispr['essentiality'] = np.log(crispr['essentiality'] * 100 + 1)
    if config.with_pam:
        pam_code = 8
    else:
        pam_code = 0
    # scale_features
    process_features.scale_features(crispr)
    process_features.scale_output(crispr)
    logger.debug("Read in data successfully")

    logger.debug("Transforming data")
    X_for = crispr.loc[:, 'sequence'].apply(lambda seq: utils.split_seqs(seq[:config.seq_len]))
    X_rev = crispr.loc[:, 'sequence'].apply(lambda seq: utils.split_seqs(seq[config.seq_len-1::-1]))
    X_cnn = crispr.loc[:, 'sequence'].apply(lambda seq: utils.split_seqs(seq[:config.seq_len], nt=1))
    X = pd.concat([X_for, X_rev, X_cnn], axis=1)
    logger.debug("Get sequence sucessfully")
    off_target_X = pd.DataFrame(np.empty(shape=[X_for.shape[0], 0]))
    # off_target_X = crispr.loc[:, 'sequence'].apply(lambda seq: utils.map_to_matrix(seq, 1, 22))
    # y = pd.DataFrame(np.abs(crispr[config.y].copy()) * 10)
    y = pd.DataFrame(crispr[config.y].copy() * 8)
    logger.debug("Transformed data successfully")

    logger.debug("Starting to prepare for splitting dataset to training dataset and testing dataset based on genes")
    logger.debug("Generating groups based on gene names")
    if config.group:
        crispr.loc[:, "group"] = pd.Categorical(crispr.loc[:, config.group])
    logger.debug("Generated groups information successfully")

    logger.debug("Splitting dataset")
    if os.path.exists(config.train_index) and os.path.exists(config.test_index):
        train_index = pickle.load(open(config.train_index, "rb"))
        test_index = pickle.load(open(config.test_index, "rb"))
    else:
        train_test_split = getattr(process_features, config.split_method+"_split", process_features.regular_split)
        train_index, test_index = train_test_split(crispr, group_col = config.group_col, n_split = max(len(crispr)/100, 10), rd_state=7)

        with open(config.train_index, 'wb') as train_file:
                pickle.dump(train_index, train_file)
        with open(config.test_index, 'wb') as test_file:
                pickle.dump(test_index, test_file)
   
    if config.test_cellline:
        test_cellline_index = crispr[crispr['cellline'] == config.test_cellline].index
        test_index = test_cellline_index.intersection(test_index)
 
    test_index_list = [x.index for _, x in crispr.loc[test_index, :].reset_index().groupby('group')
                       if len(x)] if config.test_method == 'group' else []
    logger.debug("Splitted data successfully")

    logger.debug("training data amounts: %s, testing data amounts: %s" % (len(train_index), len(test_index)))
    x_train, x_test, y_train, y_test, off_target_X_train, off_target_X_test = \
                                       X.loc[train_index, :], X.loc[test_index, :], \
                                       y.loc[train_index, :], y.loc[test_index, :], \
                                       off_target_X.loc[train_index, :], off_target_X.loc[test_index, :]



    _, unique_train_index = np.unique(pd.concat([x_train, y_train], axis=1), return_index=True, axis=0)
    _, unique_test_index = np.unique(pd.concat([x_test, y_test], axis=1), return_index=True, axis=0)
    logger.debug("after deduplication, training data amounts: %s, testing data amounts: %s" % (len(unique_train_index), len(unique_test_index)))
    logger.debug("Splitted dataset successfully")

    logger.debug("Generating one hot vector for categorical data")

    extra_crispr_df = crispr[config.extra_categorical_features + config.extra_numerical_features]

    n_values = [pam_code] + ([2] * (len(config.extra_categorical_features)-1)) if config.with_pam else [2] * len(config.extra_categorical_features)
    process_features.process_categorical_features(extra_crispr_df, n_values)
    extra_x_train, extra_x_test = extra_crispr_df.loc[train_index, :].values, extra_crispr_df.loc[test_index, :].values
    logger.debug("Generating on hot vector for categorical data successfully")

    logger.debug("Seperate forward and reverse seq")
    x_train = x_train.values
    for_input_len = config.seq_len - config.word_len + 1
    for_input, rev_input, for_cnn = x_train[:, :for_input_len], x_train[:, for_input_len: 2*for_input_len], x_train[:, 2*for_input_len:] 
    x_test = x_test.values
    for_x_test, rev_x_test, for_cnn_test = x_test[:, :for_input_len], x_test[:, for_input_len: 2*for_input_len], x_test[:, 2*for_input_len:]
    off_target_X_train = off_target_X_train.values
    off_target_X_test = off_target_X_test.values
    if not config.off_target:
        off_target_X_train, off_target_X_test = np.empty(shape=[off_target_X_train.shape[0], 0]), np.empty(shape=[off_target_X_test.shape[0],0])

    if (not config.rev_seq) or (config.model_type == 'mixed'):
        rev_input, rev_x_test = np.empty(shape=[rev_input.shape[0], 0]), np.empty(shape=[rev_x_test.shape[0], 0])

    y_train = y_train.values
    filter = y_train.flatten() > 0
    y_test = y_test.values


    if config.ml_train:

        try:
            ml_train(X, extra_crispr_df, y, train_index, test_index)

        except:
            logger.debug("Fail to use random forest")
        finally:
            h2o.cluster().shutdown()
        return


    logger.debug("Building the RNN graph")
    weight_matrix = [utils.get_weight_matrix()] if config.word2vec_weight_matrix else None
    for_seq_input = Input(shape=(for_input.shape[1],))
    rev_seq_input = Input(shape=(rev_input.shape[1],))
    for_cnn_input = Input(shape=(for_cnn.shape[1],))
    bio_features = Input(shape=(extra_x_train.shape[1],))
    off_target_features = Input(shape=(off_target_X_train.shape[1],))
    all_features = Input(shape=(for_input.shape[1]+rev_input.shape[1]+extra_x_train.shape[1]+off_target_X_train.shape[1],))
    if not config.ensemble:
        crispr_model = models.CrisprCasModel(bio_features = bio_features, for_seq_input = for_seq_input,
                                         rev_seq_input = rev_seq_input, weight_matrix = weight_matrix,
                                         off_target_features = off_target_features, all_features = all_features).get_model()
    else:
        crispr_model = models.CrisprCasModel(bio_features = bio_features, for_seq_input = for_seq_input,
                                         rev_seq_input = rev_seq_input, for_cnn_input = for_cnn_input, weight_matrix = weight_matrix,
                                         off_target_features = off_target_features, all_features = all_features).get_model()

    if config.retraining:
        loaded_model = load_model(config.retraining_model, custom_objects={'revised_mse_loss': utils.revised_mse_loss, 'tf':tf})
        for layer in loaded_model.layers:
            print(layer.name)

        
        if config.model_type == 'cnn':

            for_layer = loaded_model.get_layer(name = 'embedding_1')
            for_layer.trainable = config.fine_tune_trainable

            full_connected = loaded_model.get_layer(name='sequential_6')


        elif (config.model_type == 'mixed') or (config.model_type == 'ensemble'):

            for_layer = loaded_model.get_layer(name = 'sequential_5')
            if config.frozen_embedding_only:
                for_layer = for_layer.get_layer(name = 'embedding_1')
            for_layer.trainable = config.fine_tune_trainable

            cnn_layer = loaded_model.get_layer(name='embedding_2')
            cnn_layer.trainable = config.fine_tune_trainable
            if not config.frozen_embedding_only:
                cnn_layer_1 = loaded_model.get_layer(name='sequential_3')
                cnn_layer_2 = loaded_model.get_layer(name='sequential_4')
                cnn_layer_1.trainable = config.fine_tune_trainable
                cnn_layer_2.trainable = config.fine_tune_trainable

            full_connected = loaded_model.get_layer(name='sequential_6')

        else:
            for_layer = loaded_model.get_layer(name='sequential_5')
            if config.frozen_embedding_only:

                for_layer = for_layer.get_layer(name = 'embedding_1')
            for_layer.trainable = config.fine_tune_trainable
            if config.rev_seq:
                rev_layer = loaded_model.get_layer(name='sequential_2')
                if config.frozen_embedding_only:
                    rev_layer = rev_layer.get_layer(name = 'embedding_2')
                rev_layer.trainable = config.fine_tune_trainable
                full_connected = loaded_model.get_layer(name = 'sequential_3')
            else:
                full_connected = loaded_model.get_layer(name='sequential_6')

        for i in range(int((len(full_connected.layers)/4) * (1-config.fullly_connected_train_fraction))):

            dense_layer = full_connected.get_layer(name='dense_' + str(i+1))
            dense_layer.trainable = config.fine_tune_trainable


        crispr_model = models.CrisprCasModel.compile_transfer_learning_model(loaded_model)

    utils.output_model_info(crispr_model)
    logger.debug("Built the RNN model successfully")

    try:
        if config.training:
            logger.debug("Training the model")
            # x_train = x_train.values.astype('int32').reshape((-1, 21, 200))
            checkpoint = ModelCheckpoint(config.temp_hdf5_path, verbose=1, save_best_only=True, period=1)
            reduce_lr = LearningRateScheduler(utils.cosine_decay_lr)

            logger.debug("augmenting data")
            processed_for_input = utils.augment_data(for_input, filter=filter, is_seq=True) if config.augment_data else for_input

            if config.augment_data:
                if rev_input.shape[0] and rev_input.shape[1]:
                    processed_rev_input = utils.augment_data(rev_input, filter=filter, is_seq=True, is_rev=True)
                else:
                    processed_rev_input = utils.augment_data(rev_input, filter=filter)
            else:
                processed_rev_input = rev_input

            processed_off_target_X_train = utils.augment_data(off_target_X_train, filter=filter) if config.augment_data else off_target_X_train
            processed_extra_x_train = utils.augment_data(extra_x_train, filter=filter) if config.augment_data else extra_x_train
            processed_y_train = utils.augment_data(y_train, filter=filter) if config.augment_data else y_train
            logger.debug("augmented data successfully")

            logger.debug("selecting %d data for training" %(config.retraining_datasize*len(processed_y_train)))
            index_range = list(range(len(processed_y_train)))
            np.random.shuffle(index_range)
            selected_index = index_range[:int(config.retraining_datasize*len(processed_y_train))]
            logger.debug("selecting %d data for training" %(config.retraining_datasize*len(processed_y_train)))


            features_list = [processed_for_input[selected_index],
                                            processed_rev_input[selected_index],
                                            processed_off_target_X_train[selected_index],
                                            processed_extra_x_train[selected_index]]

            if config.ensemble:
                processed_for_cnn = utils.augment_data(for_cnn, filter=filter,
                                                       is_seq=True) if config.augment_data else for_cnn
                features_list.append(processed_for_cnn[selected_index])
                print("ensemble")
                print(len(features_list))

            training_history = utils.print_to_training_log(crispr_model.fit)(x=features_list,
                                                validation_split=0.05, y=processed_y_train[selected_index],
                                                epochs=config.n_epochs,
                                                batch_size=config.batch_size, verbose=2,
                                                callbacks=[checkpoint, reduce_lr])

            logger.debug("Saving history")
            with open(config.training_history, 'wb') as history_file:
                pickle.dump(training_history.history, history_file)
            logger.debug("Saved training history successfully")

            logger.debug("Trained crispr model successfully")

        else:
            logger.debug("Logging in old model")
            loaded_model = load_model(config.old_model_hdf5, custom_objects={'revised_mse_loss': utils.revised_mse_loss, 'tf':tf})
            crispr_model = models.CrisprCasModel.compile_transfer_learning_model(loaded_model)
            crispr_model.save(config.temp_hdf5_path)
            logger.debug("Load in model successfully")

    except KeyboardInterrupt as e:

        logger.debug("Loading model")
        loaded_model = load_model(config.temp_hdf5_path, custom_objects={'revised_mse_loss': utils.revised_mse_loss, 'tf':tf})
        crispr_model = models.CrisprCasModel.compile_transfer_learning_model(loaded_model)
        logger.debug("Load in model successfully")

    logger.debug("Persisting model")
    # serialize weights to HDF5
    crispr_model.save(config.hdf5_path)
    print("Saved model to disk")

    logger.debug("Loading best model for testing")
    loaded_model = load_model(config.temp_hdf5_path,
                              custom_objects={'revised_mse_loss': utils.revised_mse_loss, 'tf': tf})
    crispr_model = models.CrisprCasModel.compile_transfer_learning_model(loaded_model)
    logger.debug("Load in model successfully")

    logger.debug("Predicting data with best model")
    train_list = [for_input[unique_train_index], rev_input[unique_train_index],
                                                 off_target_X_train[unique_train_index], extra_x_train[unique_train_index]]
    if config.ensemble:
        train_list.append(for_cnn[unique_train_index])
    train_prediction = crispr_model.predict(x = train_list)
    train_performance = spearmanr(train_prediction, y_train[unique_train_index])
    logger.debug("GRU model spearman correlation coefficient for training dataset is: %s" % str(train_performance))

    get_prediction = getattr(sys.modules[__name__], "get_prediction_" + config.test_method, get_prediction_group)
    test_list = [for_x_test, rev_x_test, off_target_X_test, extra_x_test]
    if config.ensemble:
        test_list.append(for_cnn_test)
    performance, prediction = get_prediction(crispr_model, test_index_list, unique_test_index, y_test, test_list)
    logger.debug("GRU model spearman correlation coefficient: %s" % str(performance))

    logger.debug("Loading last model for testing")
    loaded_model = load_model(config.hdf5_path,
                              custom_objects={'revised_mse_loss': utils.revised_mse_loss, 'tf': tf})
    crispr_model = models.CrisprCasModel.compile_transfer_learning_model(loaded_model)
    logger.debug("Load in model successfully")

    logger.debug("Predicting data with last model")
    last_train_prediction = crispr_model.predict(x = train_list)
    last_train_performance = spearmanr(last_train_prediction, y_train[unique_train_index])
    utils.output_config_info()
    logger.debug("GRU model spearman correlation coefficient for training dataset is: %s" % str(last_train_performance))

    last_performance, last_prediction = get_prediction(crispr_model, test_index_list, unique_test_index, y_test, test_list)
    logger.debug("GRU model spearman correlation coefficient: %s" % str(last_performance))

    logger.debug("Saving test and prediction data plot")
    if last_performance > performance:
        prediction = last_prediction
    utils.ytest_and_prediction_output(y_test[unique_test_index], prediction)
    logger.debug("Saved test and prediction data plot successfully")

    if config.check_feature_importance:
        if performance > last_performance:
            loaded_model = load_model(config.temp_hdf5_path,custom_objects={'revised_mse_loss': utils.revised_mse_loss, 'tf': tf})
            crispr_model = models.CrisprCasModel.compile_transfer_learning_model(loaded_model)
        logger.debug("Getting features ranks")
        names = []
        names += ["for_" + str(i) for i in range(for_input.shape[1])]
        names += ["rev_" + str(i) for i in range(rev_input.shape[1])]
        names += ["off_" + str(i) for i in range(off_target_X_train.shape[1])]
        names += config.extra_categorical_features + config.extra_numerical_features
        ranker = feature_imp.InputPerturbationRank(names)
        feature_ranks = ranker.rank(20, y_test[unique_test_index], crispr_model, [data[unique_test_index] for data in test_list])
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
        logger.debug("clean directory %s" %config.run_dir)
        raise
