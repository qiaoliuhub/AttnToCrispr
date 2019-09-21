import logging
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import GroupKFold, ShuffleSplit, StratifiedKFold, GroupShuffleSplit, KFold
import numpy as np
import pandas as pd
import sys
import importlib
import random

config_path = ".".join(["models", sys.argv[1]]) + "." if len(sys.argv) >= 2 else ""
config = importlib.import_module(config_path+"config")

logging.basicConfig()
logger = logging.getLogger("features_process")
logger.setLevel(logging.DEBUG)


def scale_features(df):

    logger.debug("Scaling features")
    if len(config.scale_feature):
        scaler = MinMaxScaler()
        df.loc[:, config.scale_feature] = scaler.fit_transform(df.loc[:, config.scale_feature]) 
    logger.debug("Scale data successfully")


def scale_output(df):
    logger.debug("Scaling output")
#    scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit_transform(df[config.y])
    scaler2 = MinMaxScaler()
    scaler2.fit_transform(df[config.y])
    logger.debug("Scale data successfully")


def process_categorical_features(df, cat_values):

    # Transfer categorical features to numerical label data using LabelEncoder
    if not len(config.extra_categorical_features):
        return

    for col in config.extra_categorical_features:

        le = LabelEncoder()
        if col == 'PAM':
            train3mer = [a + b + c for a in 'ATCG' for b in 'AG' for c in 'G']
            le.fit(train3mer)
            df.loc[:, col] = le.transform(df.loc[:,col])
        df.loc[:, col] = le.fit_transform(df.loc[:,col])

    # Generate a boolean mask, only for the one needed for one hot encoder
    categorical_features = df.columns.isin(config.extra_categorical_features)
    # Generate the one hot encoder to transform categorical data
    ohe = OneHotEncoder(n_values= cat_values, categorical_features=categorical_features, handle_unknown="ignore", sparse=False)
    transformed_df = ohe.fit_transform(df)
    return transformed_df

def leave_gene_split(df, col=None, n_split = 10, rd_state = 3):

    # Group category codes
    if col is not None:
        assert col in df.columns
        df.loc[:, "group"] = pd.Categorical(df.loc[:, col]).codes

    else:
        df.loc[:, "group"] = pd.cut(df[config.y[0]], bins=10).astype('category').codes
    gkf = GroupKFold(n_splits = n_split)
    gkf_split = gkf.split(df, groups=df.loc[:, "group"])
    for i in range(rd_state):
         next(gkf_split)
    return next(gkf_split)

def leave_group_sgRNA_split(df, col = None, n_split = 5, rd_state = 3, test_size = 1/29):

    #gss = GroupShuffleSplit(n_splits=n_split, random_state=rd_state, test_size=test_size)
    gss = GroupKFold(n_splits = n_split)
    if col is not None:
        assert col in df.columns
        df.loc[:, "group"] = pd.Categorical(df.loc[:, col]).codes

    else:
        df.loc[:, "group"] = pd.cut(df[config.y[0]], bins=10).astype('category').codes

    gss_split = gss.split(np.zeros(len(df)), groups=df.loc[:, "group"])
    rd_state = random.choice(range(10))
    for i in range(rd_state):
        tr, te = next(gss_split)
        print(tr, te)
    print(rd_state)
    return next(gss_split)

def leave_off_target_sgRNA_split(df, col = None, n_split = 10, rd_state = 3, test_size = 1/3):

    gss = GroupKFold(n_splits=n_split)
    if col is not None:
        assert col in df.columns
        df.loc[:, "group"] = pd.Categorical(df.loc[:, col]).codes

    else:
        df.loc[:, "group"] = pd.cut(df[config.y[0]], bins=10).astype('category').codes

    group_1_filter = (df.loc[:,'class'].astype(int) == 1)
    group_0_filter = (df.loc[:,'class'].astype(int) == 0)
    group_1, group_0 = df[group_1_filter], df[group_0_filter]
    gss_split = gss.split(df, groups=df.loc[:, "group"])
    for i in range(rd_state % n_split):
        next(gss_split)
    train_gss, test_gss = next(gss_split)
    shuffle_split = ShuffleSplit(test_size=test_size, random_state=rd_state)
    sf_split = shuffle_split.split(df)
    for _ in range(rd_state%n_split):
        next(sf_split)
    train_sf, test_sf = next(sf_split)
    train_index = (set(train_gss) & set(group_1.index)) | (set(train_sf) & set(group_0.index))
    test_index = (set(test_gss) & set(group_1.index)) | (set(test_sf) & set(group_0.index))
    return np.array(list(train_index)), np.array(list(test_index))

def regular_split(df, col=None, n_split = 10, rd_state = 3):

    shuffle_split = KFold(n_splits=n_split, random_state=42, shuffle=True)
    #shuffle_split = ShuffleSplit(test_size=1.0/n_split, random_state = rd_state)
    sf_split = shuffle_split.split(df)
    for _ in range(rd_state%n_split):
        next(sf_split)
    return next(sf_split)

def stratified_split(df, col=None, n_split = 10, rd_state = 3):

    skf = StratifiedKFold(n_splits=n_split, shuffle = True, random_state= rd_state)
    if col is not None:
        assert col in df.columns
        df.loc[:, "group"] = pd.Categorical(df.loc[:, col]).codes

    else:
        df.loc[:, "group"] = pd.cut(df[config.y[0]], bins=10).astype('category').codes

    logger.debug("Generated groups information successfully")

    skf_split = skf.split(np.zeros(len(df)), df.loc[:, "group"])
    for i in range(rd_state%n_split):
        next(skf_split)
    return next(skf_split)
