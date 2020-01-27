import config
import os

output_FF_layers = [100, 1] #[4000, 1000, 100, 1] #[200, 200, 100, 100, 1]
cur_work_dir = os.getcwd()
d_model = 64
attention_heads = 32
attention_dropout = 0.0
n_layers = 1
k_dim = 10
add_seq_cnn = True
add_parallel_cnn = False
cnn_dropout = 0.2
attention_norm = False
attention_layer_norm = False
n_feature_dim = config.embedding_vec_dim
analysis = None #'deepCrispr'
oversample = True
#(should always be 1)

data_folder = os.path.join(cur_work_dir, 'datas_cpf1')
if not os.path.exists(data_folder):
    print("Create {0} directory".format(data_folder))
    os.mkdir(data_folder)
