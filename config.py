import os
from datetime import datetime
import shutil
import sys
from pytz import timezone

# cellline folder, test, K562, A549, HUH7, NB4 or SKNMC
if len(sys.argv) >= 2 and not sys.argv[1].split("/")[-1].startswith("run"):
    cellline = sys.argv[1].split("/")[-1]
elif len(sys.argv) >= 2:
    cellline = sys.argv[1].split("/")[-2]
else:
    cellline = "test"

# data speficic name, for example, if file name is K562_CN, then data_specific is _CN
data_specific = "_CN_centrality" if cellline != "test" else ""
update_features = False

# sequence length
seq_start = 2
sep_len = 0
seq_len = 20

# model types: concat, mul, mixed, ensemble
model_type = "ensemble"
seq_cnn = True
ensemble = (model_type == "ensemble")

# epochs
n_epochs = 5
patience = 100
decay_steps = 600
min_lr = 0.005

# model hyperparameters
start_lr = 0.0005
dropout = 0.1
batch_size = 50
lr_decay = 0.001
maxnorm = 5
augment_data = False
#with_pam = False
log_cen = False
cnn_levels = [32, 64]
fully_connected_layer_layout = [128] *1 + [64] * 1 
activation_method =["swish"]
bio_fully_connected_layer_layout = [8]
LSTM_stacks_num = 2
LSTM_hidden_unit = 64
rnn_time_distributed_dim = 8
embedding_voca_size = 16
embedding_vec_dim = 8
word_len = 2
test_cellline = None
train_cellline = None
add_norm = False

# split method, it could be leave_gene, stratified, leave_off_target_sgRNA, leave_group_sgRNA or regular
split_method = "regular"
# could be "regular", "group"
test_method = "regular"
# categorize gene symbols or None for split or test use
group = 'sequence' #"Chromosome" #"symbol", "log2fc"
# could be None and "cellline", which was used by stratified split method
group_col = None


# deco_part of output directory, run_ + deco_part + time
deco_part = "_".join([cellline, data_specific, split_method[:3], model_type[:3]])

# weight matrix cold start
word2vec_weight_matrix = False

# MIN_MAX scale festure
#centrality_features = ["Degree","Closeness", "Betweennes", "NetworkConstraint",
                       #"ClusteringCoefficient", "PageRank"]
scale_feature = [] #+ ["r_d_tm" ] + ['aa_cut'] # + ['essentiality']# + ['log_CN']#+ ["centrality"] #+ ['aa_cut'] #["log_count", "r_d_tm", "log_CN"] + ["centrality"]

# RNN input features, every unit input is 200 elements vector
seq2vec_features = ["sequence"]

# outlayer extra features
extra_numerical_features = [] #+ ["r_d_tm" ] + ['aa_cut'] #+ ['essentiality'] #+ ['log_CN']# + ["centrality"]# + ['aa_cut'] #+ ["aa_cut", "r_d_tm", "log_count","log_CN"] + ["centrality"]
extra_categorical_features = [] #, ['epi_' + str(i) for i in range(4)] #+ ["dnase_sensitivity"] # + ['edge']

# if with_pam:
#     extra_categorical_features = ["PAM"] + extra_categorical_features
off_target = False
rev_seq = False

# output
y = ['log2fc'] # ["Normalized efficacy"] #'log2fc'
y_transform = True
y_inverse_transform = True
log2fc_filter = False

# current directory
cur_dir = os.getcwd()

# Whether check feature importance using deep learning
check_feature_importance = True

# Whether to do RF training
ml_train = False

# Whether or not to train the model again
training = True
old_model_hdf5_path = os.path.join(cur_dir, "dataset/K562/run_K562__CN_centrality_reg_ens1904240007")
old_model_hdf5 = os.path.join(old_model_hdf5_path, "lstm_model.h5")

# parameters when retraining
retraining = False
fine_tune_trainable = True
retraining_datasize = 3.0/3 if retraining else 1
frozen_embedding_only = True
fullly_connected_train_fraction = 1.0/1
retraining_model_path = os.path.join(cur_dir, "dataset/run_cpf1__CN_centrality_reg_mix1811071353")
retraining_model = os.path.join(retraining_model_path, "attn_model.h5")
retraining_model_state = os.path.join(retraining_model_path, "attn_model_state.h5")
retraining_dataset = ""


# each run specific
rerun_name = sys.argv[1].split("/")[-1] if len(sys.argv) >= 2 and sys.argv[1].split("/")[-1].startswith("run") else None
unique_name = rerun_name if rerun_name else "run_" + deco_part + datetime.now(timezone('US/Eastern')).strftime("%y%m%d%H%M")

data_dir = os.path.join(cur_dir, "dataset", cellline)
run_dir = os.path.join(data_dir, unique_name)

# pretrained seq2vec 3mer vector representation
seq2vec_mapping = os.path.join(cur_dir, "dataset", "cds_vector_"+str(embedding_vec_dim))
mismatch_matrix = os.path.join(cur_dir, "dataset", "off_target_matrix.csv")
pam_matrix = os.path.join(cur_dir, "dataset", "Pam_score.csv")

# Directory of input dataset, /work/qliu_data/crispr_data/crispr/RNN_with_seq2vec/crispr_with_features_K562.csv
#input_dataset = "/work/qliu_data/crispr_data/crispr/crispr_with_features/crispr_with_features_K562_cas9.csv"
input_dataset = os.path.join(data_dir, "{!s}{!s}.csv".format(cellline, data_specific))
if retraining and retraining_dataset != "":
    input_dataset = retraining_dataset

# training and test index
train_index = os.path.join(data_dir, "train_index" + data_specific)
test_index = os.path.join(data_dir, "test_index" + data_specific)
if retraining:
    train_index += "_retrain"
    test_index += "_retrain"

# if execute with a new dataset, use the config.py in cur working directory, create a new
# directory for this execution, create __init__ to generate a package, then copy
# cur working directory to the new created directory
#
# else use the existed config file
run_specific_config = os.path.join(run_dir, "config.py")
cur_dir_config = os.path.join(cur_dir, "config.py")
run_specific_attention_setting = os.path.join(run_dir, "attention_setting.py")
cur_dir_attention_setting = os.path.join(cur_dir, "attention_setting.py")

# save run specific data
retransform = False
run_specific_data_after_transform = os.path.join(data_dir, "after_transform_data")


if not os.path.exists(run_dir):
    os.makedirs(run_dir)
    open(os.path.join(data_dir, "__init__.py"), 'w+').close()
    open(os.path.join(run_dir, "__init__.py"), 'w+').close()
    shutil.copyfile(cur_dir_config, run_specific_config)
    shutil.copyfile(cur_dir_attention_setting, run_specific_attention_setting)

# Directory to save test and prediction y plot
test_prediction = os.path.join(run_dir, "test_prediction.csv")

# Directory to save the pickled training history
training_history = os.path.join(run_dir, "training_history.pi")

# model saving in hdf5 directory
hdf5_path = os.path.join(run_dir, "attn_model.h5")
hdf5_path_state = os.path.join(run_dir, "attn_model_state.h5")

# machine learning model saving path
ml_model_path = os.path.join(run_dir, "ml_model")
feature_importance_path = os.path.join(run_dir, 'features.csv')

# temprature save models
temp_hdf5_path = os.path.join(run_dir, "temp_model.h5")

# directory to save logfile
run_specific_log = os.path.join(run_dir, "logfile")

# logfile to save the training process
training_log = os.path.join(run_dir, "training_log")

# logfile to save the file used for tensorboard
tb_log = os.path.join(run_dir, "tb_log")

# output_attrs list
output_attrs = ["split_method", "model_type", "dropout", "scale_feature", "seq2vec_features", "extra_numerical_features",
                "extra_categorical_features", "cellline", "word2vec_weight_matrix", "input_dataset"]






