import os
from torch import cuda, device
from torch import save
import torch
import sys
import importlib
import logging
import pandas as pd
import process_features
import utils
import attention_setting
import pickle
from skorch import NeuralNetClassifier, NeuralNetRegressor
import attention_model
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, GroupShuffleSplit, GroupKFold, KFold
from imblearn.over_sampling import RandomOverSampler
import feature_imp
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
import pdb

### setting pytorch working environment
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
use_cuda = cuda.is_available()
if use_cuda:
    device2 = device("cuda:0")
    cuda.set_device(device2)
    cuda.empty_cache()
else:
    device2 = device("cpu")
torch.set_default_tensor_type("torch.FloatTensor")

# Setting the correct config file
config_path = ".".join(["models", sys.argv[1]]) + "." if len(sys.argv) >= 2 else ""
config = importlib.import_module(config_path + "config")
attention_setting = importlib.import_module(config_path+"attention_setting")

# Setting up log file
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)s %(name)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh = logging.FileHandler(config.run_specific_log, mode='a')
fh.setFormatter(fmt=formatter)
logger = logging.getLogger("Crispr off-target")
logger.addHandler(fh)
logger.setLevel(logging.DEBUG)

class data_preparer:

    crispr = None
    def __init__(self, add_pam = False, scale_feature = False, trg_seq_col = None):
        logger.debug("Reading in the crispr dataset %s" % config.input_dataset)
        self.initialize_crispr()
        if scale_feature:
            process_features.scale_features(self.crispr)
        if config.group:
            self.generate_group_split_col()
        logger.debug("Read in data successfully")
        self.add_pam = add_pam
        self.trg_seq_col = trg_seq_col
        self.feature_length_map = []
        self.X = None

    @property
    def feature_length_map(self):
        assert len(self.__feature_length_map) == 3
        return self.__feature_length_map

    @feature_length_map.setter
    def feature_length_map(self, val):
        if isinstance(val, list) and (len(val) == 0 or len(val) == 3):
            self.__feature_length_map = val
        else:
            logger.error("feature length map has to be assigned as a list")
            self.__feature_length_map = []

    @classmethod
    def initialize_crispr(cls):
        if cls.crispr is None:
            cls.crispr = pd.read_csv(config.input_dataset)
            assert 'sequence' in cls.crispr.columns, "no sequence columns was found in crispr database"

    def prepare_x(self, mismatch = False, trg_seq_col = None):

        if self.X is not None:
            return self.X

        if mismatch:
            assert trg_seq_col is not None
            X_src, src_len = self.generate_splitted_mismatch('sequence', trg_seq_col)
            cur_index = 0
            self.__feature_length_map.append((cur_index, cur_index+src_len))
            cur_index += src_len
            self.__feature_length_map.append(None)

        else:
            X_src, src_len = self.generate_splitted_nts()
            cur_index = 0
            self.__feature_length_map.append((cur_index, cur_index+src_len))
            cur_index += src_len

            if trg_seq_col:
                X_trg, trg_len = self.generate_splitted_nts(trg_seq_col)
                self.__feature_length_map.append((cur_index, cur_index+trg_len))
                cur_index += cur_index + trg_len
                X_src = pd.concat([X_src, X_trg], axis=1)
            else:
                self.__feature_length_map.append(None)

        extra_feature_len = len(config.extra_categorical_features + config.extra_numerical_features)
        if extra_feature_len:
            if attention_setting.analysis == 'deepCrispr':
                epis = []
                for fea in config.extra_categorical_features + config.extra_numerical_features:
                    epis.append(self.crispr[fea].apply(lambda x: pd.Series(list(x)[:src_len]).astype(int)))
                X_src = pd.concat([X_src] + epis, axis=1)
                self.__feature_length_map.append((cur_index, cur_index + 4*src_len))
                cur_index += 4*src_len
            else:
                extra_crispr_df, extra_feature_len_with_cat = self.get_extra_feature()
                X_src = pd.concat([X_src, extra_crispr_df], axis=1)
                self.__feature_length_map.append((cur_index, cur_index + extra_feature_len_with_cat))
                cur_index += extra_feature_len_with_cat
        else:
            self.__feature_length_map.append(None)

        self.X = X_src
        return X_src

    def persist_data(self):
        logger.debug("persisting data ...")
        if self.X is None:
            self.X = self.prepare_x(self.trg_seq_col)
        for i, combin_drug_feature_array in enumerate(self.X.values):
            if config.update_features or not os.path.exists(
                    os.path.join(attention_setting.data_folder, str(i) + '.pt')):
                save(combin_drug_feature_array, os.path.join(attention_setting.data_folder, str(i) + '.pt'))

    @classmethod
    def get_crispr_preview(cls):
        if cls.crispr is not None:
            return cls.crispr.head()
        else:
            return

    def get_pam(self, start = None, length = None):

        if 'PAM' in self.crispr.columns:
            return self.crispr['PAM']
        elif start is not None and length is not None:
            return self.crispr['sequence'].str[start: start+length]
        else:
            logger.error("No pam are found")
            return

    def generate_splitted_nts(self, seq_column = 'sequence'):

        saved_file_name = config.run_specific_data_after_transform + "_" + seq_column.replace(" ", "")
        if config.retransform or (not os.path.exists(saved_file_name)):
            nts = self.crispr.loc[:, seq_column].apply(
                lambda seq: utils.split_seqs(seq[config.seq_start:config.seq_start + config.seq_len]))
            pickle.dump(nts, open(saved_file_name, "wb"))
        else:
            nts = pickle.load(open(saved_file_name, "rb"))
        logger.debug("Split sequence to pieces successfully")
        return nts, config.seq_len - config.word_len + 1

    def generate_splitted_mismatch(self, RNA, DNA):

        saved_file_name = config.run_specific_data_after_transform + "_mismatch"
        if config.retransform or (not os.path.exists(saved_file_name)):
            nts = self.crispr.apply(
                lambda row: utils.split_mismatchs(row[RNA], row[DNA]), axis = 1)
            pickle.dump(nts, open(saved_file_name, "wb"))
        else:
            nts = pickle.load(open(saved_file_name, "rb"))
        logger.debug("Split sequence to pieces successfully")
        return nts, config.seq_len - config.seq_start

    def get_extra_feature(self):

        extra_crispr_df = self.crispr[config.extra_categorical_features + config.extra_numerical_features]
        n_values = [2] * len(config.extra_categorical_features)
        process_features.process_categorical_features(extra_crispr_df, n_values)
        logger.debug("Generating one hot vector for categorical data successfully")
        return extra_crispr_df, extra_crispr_df.shape[1]

    def get_labels(self, binary = False):
        if binary:
            return (self.crispr[config.y] > 0).astype(int)
        return self.crispr[config.y]

    def generate_group_split_col(self, col = config.group):

        assert col in self.crispr.columns
        self.crispr.loc[:, "group"] = pd.Categorical(self.crispr.loc[:, col]).codes
        logger.debug("Generated groups information successfully")
        return self.crispr.loc[:, "group"].astype(int)

    def train_test_split(self, n_split = None):
        logger.debug("Splitting dataset")
        if os.path.exists(config.train_index) and os.path.exists(config.test_index):
            try:
                train_index = pickle.load(open(config.train_index, "rb"))
                test_index = pickle.load(open(config.test_index, "rb"))
            except UnicodeDecodeError as e:
                train_index = pickle.load(open(config.train_index, "rb"), encoding='latin1')
                test_index = pickle.load(open(config.test_index, "rb"), encoding='latin1')
            except:
                raise
        else:
            train_test_split = getattr(process_features, config.split_method + "_split", process_features.regular_split)
            if n_split is None:
                n_split = max(len(self.crispr) // 10000, 10)
            train_index, test_index = train_test_split(self.crispr, col = config.group,
                                                       n_split=n_split, rd_state=7)

            with open(config.train_index, 'wb') as train_file:
                pickle.dump(train_index, train_file)
            with open(config.test_index, 'wb') as test_file:
                pickle.dump(test_index, test_file)

        return train_index, test_index


def numerical_to_class_metric(y_true, y_pred):

    from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, recall_score, f1_score
    cut_off = np.quantile(y_true, 0.8)
    y_true = (y_true > cut_off).astype(int)
    pred_cut_off = np.quantile(y_pred, 0.8)
    y_pred_binary = (y_pred > pred_cut_off).astype(int)
    #y_pred = 1 / (1 + np.exp(-y_pred))
    auc = roc_auc_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = average_precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    return auc

def spearman(y_true, y_pred):

    from scipy.stats import spearmanr
    return spearmanr(y_true.reshape(-1), y_pred.reshape(-1))[0]

def pearson(y_true, y_pred):

    from scipy.stats import pearsonr
    return pearsonr(y_true.reshape(-1), y_pred.reshape(-1))[0]

def classifier_training(crispr_model_classifier, X, y_binary, cv_splitter):

    split_iter = [ls for ls in cv_splitter]
    net_classifer = NeuralNetClassifier(crispr_model_classifier,
                                        optimizer=torch.optim.Adam,
                                        lr=config.start_lr,
                                        optimizer__weight_decay=config.lr_decay,
                                        optimizer__betas=(0.9, 0.98),
                                        optimizer__eps=1e-9,
                                        batch_size=config.batch_size,
                                        max_epochs=config.n_epochs,
                                        device=device2)
    net_classifer = RandomForestClassifier(n_estimators = 30)

    cv_results = cross_validate(net_classifer, X, y_binary, scoring=['roc_auc', 'average_precision'],
                                cv=split_iter, return_estimator=True, verbose=0)

    new_cv_splitter = iter(split_iter)
    results_dfs = []
    last_train = 0
    for i in range(5):
        cur_train, cur_test = next(new_cv_splitter)
        if i == 0:
            last_train = cur_train
        y_true = y_binary[cur_test]
        y_pred = cv_results['estimator'][i].predict_proba(X[cur_test, :])[:, 1]
        result_df = pd.DataFrame({'ground_truth': y_true, 'prediction': y_pred, 'fold': i})
        results_dfs.append(result_df)

    results_df = pd.concat(results_dfs, ignore_index=True)
    results_df.to_csv(config.test_prediction, index=False, mode='a+')
    logger.debug("{0!r}".format(cv_results['test_roc_auc']))
    logger.debug("{0!r}".format(cv_results['test_average_precision']))
    logger.debug("{0!r}".format(cv_results.keys()))
    return cv_results['estimator'][0], last_train

def regressor_training(crispr_model_regressor, X, y, cv_splitter_reg):
    net = NeuralNetRegressor(crispr_model_regressor,
                             optimizer=torch.optim.Adam,
                             lr=config.start_lr,
                             optimizer__weight_decay=config.lr_decay,
                             optimizer__betas=(0.9, 0.98),
                             optimizer__eps=1e-9,
                             batch_size=config.batch_size,
                             max_epochs=config.n_epochs,
                             device=device2)

    net = RandomForestRegressor(n_estimators=30)
    cv_results_reg = cross_validate(net, X, y,
                                    scoring={'spearman': make_scorer(spearman), 'pearson': make_scorer(pearson), 'neg_mean_squared_error': 'neg_mean_squared_error'},
                                    cv=cv_splitter_reg, return_estimator=True)
    logger.debug("{0!r}".format(cv_results_reg['test_spearman']))
    logger.debug("{0!r}".format(cv_results_reg['test_pearson']))
    logger.debug("{0!r}".format(cv_results_reg['test_neg_mean_squared_error']))
    logger.debug("{0!r}".format(cv_results_reg.keys()))


if __name__ == "__main__":

    data_pre = data_preparer()
    print(data_pre.get_crispr_preview())
    #X = data_pre.prepare_x(mismatch=True, trg_seq_col = 'Target sequence')
    X = data_pre.prepare_x()
    y = data_pre.get_labels()
    print(X.head())
    print(data_pre.feature_length_map)
    train_index, test_index = data_pre.train_test_split()
    print(train_index, test_index)
    torch.manual_seed(0)
    X = X.values.astype(np.float32)
    y = y.values.astype(np.float32)
    y_binary = (y > np.quantile(y, 0.8)).astype(int).reshape(-1,)
    #skf = GroupKFold(n_splits=3)
    #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_splitter = skf.split(X, data_pre.generate_group_split_col())
    if attention_setting.analysis == 'deepCrispr':
        y_binary = (y > 0).astype(int).reshape(-1,)
        crispr_model_classifier = attention_model.get_OT_model(data_pre.feature_length_map, classifier=True)
        for train_index, test_index in cv_splitter:
            X_train, X_test, y_train, y_test = X[train_index], X[test_index], y_binary[train_index], y_binary[test_index]
            if attention_setting.oversample:
                ros = RandomOverSampler(random_state=42)
                X_train, y_train = ros.fit_resample(X_train, y_train)
                train_index = ros.sample_indices_
            gss = GroupShuffleSplit(n_splits=5, random_state=42)
            cv_splitter_seq = gss.split(X_train, y_train, groups=data_pre.generate_group_split_col(col='Target sequence')[train_index])
            classifier_training(crispr_model_classifier, X_train, y_train, cv_splitter_seq)
            # cv_splitter_train = skf.split(X_train, data_pre.generate_group_split_col()[train_index])
            # for train_index, validation_index in cv_splitter_train:
            #     X_train, X_val, y_train, y_val = X[train_index], X[validation_index], y_binary[train_index], y_binary[
            #         validation_index]
            #     ros = RandomOverSampler(random_state=42)
            #     X_res, y_res = ros.fit_resample(X_train, y_train)
            #     net_classifer = NeuralNetClassifier(crispr_model_classifier,
            #                                         optimizer=torch.optim.Adam,
            #                                         lr=config.start_lr,
            #                                         optimizer__weight_decay=config.lr_decay,
            #                                         optimizer__betas=(0.9, 0.98),
            #                                         optimizer__eps=1e-9,
            #                                         batch_size=config.batch_size,
            #                                         max_epochs=config.n_epochs,
            #                                         device=device2)



    elif attention_setting.output_FF_layers[-1] == 1:
        crispr_model = attention_model.get_OT_model(data_pre.feature_length_map)
        regressor_training(crispr_model, X, y, cv_splitter)
    else:
        crispr_model_classifier = attention_model.get_OT_model(data_pre.feature_length_map, classifier=True)
        best_crispr_model, train_index = classifier_training(crispr_model_classifier, X, y_binary, cv_splitter)
    if config.check_feature_importance:

        logger.debug("Getting features ranks")
        names = []
        names += ["src_" + str(i) for i in range(data_pre.feature_length_map[0][1])]
        if data_pre.feature_length_map[1] is not None: names += ["trg_" + str(i)
                                                                 for i in range(
                data_pre.feature_length_map[1][1] - data_pre.feature_length_map[1][0])]
        if data_pre.feature_length_map[
            2] is not None: names += config.extra_categorical_features + config.extra_numerical_features
        ranker = feature_imp.InputPerturbationRank(names)
        feature_ranks = ranker.rank(2, y_binary[train_index], best_crispr_model,
                                    [X[train_index, :]])
        feature_ranks_df = pd.DataFrame(feature_ranks)
        feature_ranks_df.to_csv(config.feature_importance_path, index=False)
        logger.debug("Get features ranks successfully")






