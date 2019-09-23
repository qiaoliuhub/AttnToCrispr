import numpy as np
import pandas as pd
import tensorflow as tf
from keras.losses import mse
from keras.preprocessing.text import Tokenizer
import sys
import importlib

config_path = ".".join(["models", sys.argv[1]]) + "." if len(sys.argv) >= 2 else ""
config = importlib.import_module(config_path+"config")
attention_setting = importlib.import_module(config_path+"attention_setting")
pam_mapping = {}


def get_map(row):
    pam_mapping[row[0]] = row[1]


class __NtsTokenizer(Tokenizer):

    def __init__(self, nt):

        Tokenizer.__init__(self)
        if nt == 4:
            self.dic = [a + b + c + d for a in 'ATCG' for b in 'ATCG' for c in 'ATCG' for d in 'ATCG']
        elif nt == 3:
            self.dic = [a + b + c for a in 'ATCG' for b in 'ATCG' for c in 'ATCG']
        elif nt == 2:
            self.dic = [a + b for a in 'ATCG' for b in 'ATCG']
        elif nt == 1:
            self.dic = [a for a in 'ATCG']
        else:
            self.dic = []
        self.fit_on_texts(self.dic)



def split_seqs(seq, nt = config.word_len):

    t = __NtsTokenizer(nt = nt)

    result = ''

    lens = len(seq)
    for i in range(lens - nt + 1):
        result += ' ' + seq[i:i+nt].upper()

    seq_result = t.texts_to_sequences([result])
    return pd.Series(seq_result[0]) - 1

def split_mismatchs(seq1, seq2):

    t = __NtsTokenizer(nt=2)

    result = ''
    lens = len(seq1)
    for i in range(lens):
        result += ' ' + seq1[i] + seq2[i]

    seq_result = t.texts_to_sequences([result.upper()])
    return pd.Series(seq_result[0]) - 1

def __get_expand_table_3(rev = False):

    possibilities = pd.Series([a + b + c + d for a in 'ATCG' for b in 'ATCG' for c in 'ATCG' for d in 'ATCG']).to_frame(
        name='ori_seq')
    possibilities['key'] = 0
    change = pd.Series([a + b for a in 'ATCG' for b in 'ATCG']).to_frame(name='change')
    change['key'] = 0
    merged = pd.merge(possibilities, change, on='key')
    if rev:
        merged['new_seq'] = merged['ori_seq'].str[:2] + merged['change']
    else:
        merged['new_seq'] = merged['change'] + merged['ori_seq'].str[-2:]
    new_map = pd.Series([a + b + c + d for a in 'ATCG' for b in 'ATCG' for c in 'ATCG' for d in 'ATCG']).to_frame(
        name='new_seq')
    new_map[['pos_0', 'pos_1']] = new_map['new_seq'].apply(lambda x: split_seqs(x))
    mer_pos = pd.merge(merged, new_map, on='new_seq')
    final = pd.merge(mer_pos, new_map, left_on='ori_seq', right_on='new_seq', suffixes=('_new', '_ori'))
    final = final[['ori_seq', 'new_seq_new', 'pos_0_new', 'pos_1_new', 'pos_0_ori', 'pos_1_ori']]
    return final

def __get_expand_table_1(rev = False):

    possibilities = pd.Series([a + b for a in 'ATCG' for b in 'ATCG']).to_frame(name='ori_seq')
    possibilities['key'] = 0
    possibilities[["pos_0_ori", "pos_1_ori"]] = possibilities['ori_seq'].apply(lambda x:split_seqs(x))
    change = pd.Series([a + b for a in 'ATCG' for b in 'ATCG']).to_frame(name='new_seq_new')
    change['key'] = 0
    change[['pos_0_new', 'pos_1_new']] = change['new_seq_new'].apply(lambda x: split_seqs(x))
    merged = pd.merge(possibilities, change, on='key')[['ori_seq', 'new_seq_new', 'pos_0_new', 'pos_1_new', 'pos_0_ori', 'pos_1_ori']]
    return merged


def augment_data(df, filter, is_seq = False, is_rev = False):

    df1 = df.copy()
    filtered_df1 = df1[filter]
    kept_df1 = df1[1-filter]
    if is_seq:

        lens = filtered_df1.shape[1]
        __get_expand_table = getattr(sys.modules[__name__], "__get_expand_table_" + str(config.word_len), __get_expand_table_1)
        print(__get_expand_table)
        if is_rev:
            expand_table = __get_expand_table(rev=True)
            filtered_df1 = pd.DataFrame(filtered_df1).reset_index()
            new_df = pd.merge(filtered_df1, expand_table, left_on=[lens-3, lens-2], right_on=['pos_0_ori', 'pos_1_ori'])\
                .sort_values(by=['index'])
            return np.concatenate((kept_df1, new_df[range(lens-2) + ['pos_0_new', 'pos_1_new']].values), axis=0)

        else:
            expand_table = __get_expand_table()
            filtered_df1 = pd.DataFrame(filtered_df1).reset_index()
            new_df = pd.merge(filtered_df1, expand_table, left_on=[0, 1], right_on=['pos_0_ori', 'pos_1_ori'])\
                .sort_values(by=['index'])
            return np.concatenate((kept_df1, new_df[['pos_0_new', 'pos_1_new'] + range(2, lens)].values), axis=0)

    else:

        return np.concatenate((kept_df1, filtered_df1.repeat(16, 0)), axis=0)


def map_to_matrix(seq, start, pam_start):


    nt_ls = {'A':0, 'T':1, 'C':2, 'G':3}
    new_seq = []
    for i in range(20):
        nt = nt_ls[seq[start - 1 + i]]
        new_seq.extend(list(raw_map.loc[nt*4:(nt+1)*4-1,i]))

    new_seq.append(pam_mapping[seq[pam_start-1:pam_start+1]])
    return pd.Series(new_seq)


def get_weight_matrix():
    # get the seq2vec pre-trained vector representation of 3-mer
    embedding_index = {}
    t = __NtsTokenizer(nt = config.word_len)
    with open(config.seq2vec_mapping, 'r') as seq2vec_map:
        for line in seq2vec_map:
            data = line.split()
            trimer = data[0].lower()
            vector = np.asarray(data[1:], dtype='float32')
            embedding_index[trimer] = vector

    weight_matrix = np.zeros((config.embedding_voca_size, config.embedding_vec_dim))
    for word, index in t.word_index.items():
        embedding_vector = embedding_index[word]
        if embedding_vector is not None:
            weight_matrix[index] = embedding_vector

    return weight_matrix


def revised_mse_loss(y_true, y_pred):

    alpha = 0.9
    mse_result = mse(y_true, y_pred)
    large_coefficient = tf.where(tf.abs(y_true)<5, tf.fill(tf.shape(y_true), 0.0), tf.fill(tf.shape(y_true), 1.0))

    coefficient = tf.multiply(alpha, large_coefficient) + tf.multiply(1.0-alpha, 1.0-large_coefficient)
    result = tf.multiply(mse_result, coefficient)
    return result


def ytest_and_prediction_output(y_test, y_prediction):

    if isinstance(y_test, np.ndarray):
        y_test = pd.DataFrame(y_test)
    if isinstance(y_prediction, np.ndarray):
        y_prediction = pd.DataFrame(y_prediction.reshape(-1,))
    y_prediction.index = y_test.index
    test_prediction = pd.concat([y_test, y_prediction], axis=1)
    test_prediction.columns = ["ground_truth", "prediction"]
    test_prediction.to_csv(config.test_prediction)


def print_to_logfile(fun):

    def inner(*args, **kwargs):
        old_stdout = sys.stdout
        logfile = open(config.run_specific_log, 'a+')
        sys.stdout = logfile
        result = fun(*args, **kwargs)
        sys.stdout = old_stdout
        logfile.close()
        return result

    return inner


def print_to_training_log(fun):

    def inner(*args, **kwargs):

        old_stdout = sys.stdout
        logfile = open(config.training_log, 'a+')
        sys.stdout = logfile
        result = fun(*args, **kwargs)
        sys.stdout = old_stdout
        logfile.close()
        return result

    return inner

def cosine_decay_lr(epoch):

    global_step = min(epoch, config.decay_steps)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * global_step / config.decay_steps))
    decayed = (1 - config.min_lr) * cosine_decay + config.min_lr
    decayed_learning_rate = config.start_lr * decayed
    return decayed_learning_rate


@print_to_training_log
def output_config_info():
    print("\n".join([(attr.ljust(40) + str(getattr(config, attr))) for attr in config.output_attrs]))


@print_to_training_log
def output_model_info(model):
    model.summary()



