import config
import os

cur_work_dir = os.getcwd()
add_seq_cnn = True
add_parallel_cnn = False
output_FF_layers = [200, 1]
d_model = 20
k_dim = 10
attention_heads = 4
attention_dropout = 0.2
cnn_dropout = 0.2
attention_norm = False
attention_layer_norm = False
n_layers = 1
n_feature_dim = config.embedding_vec_dim
analysis = None#'deepCrispr' #'deepCrispr' #used in both crispr attn (add extra features) and OT_crispr_attn
oversample = True
#(should always be 1)

data_folder = os.path.join(cur_work_dir, 'datas')
if not os.path.exists(data_folder):
    print("Create {0} directory".format(data_folder))
    os.mkdir(data_folder)
