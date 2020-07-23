import torch.nn as nn
from torch import cat, transpose
import torch
import torch.nn.functional as F
from Layers import EncoderLayer, DecoderLayer
from Sublayers import Norm, OutputFeedForward
import copy
import attention_setting
import numpy as np
import crispr_attn
import math
import OT_crispr_attn
import sys
import importlib
import pdb
# Setting the correct config file
config_path = ".".join(["models", sys.argv[1]]) + "." if len(sys.argv) >= 2 else ""
config = importlib.import_module(config_path + "config")
attention_setting = importlib.import_module(config_path+"attention_setting")


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_input, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(EncoderLayer(d_input, d_model, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        x = src
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x) if attention_setting.attention_layer_norm else x

class Decoder(nn.Module):
    def __init__(self, d_input, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.layers = get_clones(DecoderLayer(d_input, d_model, heads, dropout), N)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, trg, e_outputs, src_mask=None, trg_mask=None):
        x = trg
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, src_mask, trg_mask)
        return self.norm(x) if attention_setting.attention_layer_norm else x


class Transformer(nn.Module):
    def __init__(self, d_input, d_model, n_feature_dim, N, heads, dropout, extra_length):
        super().__init__()
        self.encoder = Encoder(n_feature_dim, d_model, N, heads, dropout)
        self.decoder = Decoder(n_feature_dim, d_model, N, heads, dropout)
        #self.linear = nn.Linear()
        self.cnn = customized_CNN()
        assert not attention_setting.add_seq_cnn or not attention_setting.add_parallel_cnn
        if attention_setting.add_seq_cnn:
            d_input = 64 * (((d_input + 2) // 2 + 2) // 2)
            if attention_setting.analysis == 'deepCrispr':
                d_model += 4
                extra_length = 0
        if attention_setting.add_parallel_cnn:
            d_input_1 = d_input * d_model
            d_input_2 = ((64 * (((d_input + 2) // 2 + 2) // 2)) * config.embedding_vec_dim)
            d_input = d_input_1 + d_input_2
            d_model = 1
        self.out = OutputFeedForward(d_model, d_input, extra_length, d_layers=attention_setting.output_FF_layers, dropout=dropout)

    def forward(self, src, trg, extra_input_for_FF=None, src_mask=None, trg_mask=None):

        e_outputs = self.encoder(src, src_mask)
        # print("DECODER")
        d_output = self.decoder(trg, e_outputs, src_mask, trg_mask)
        if attention_setting.add_seq_cnn:
            if extra_input_for_FF is not None and attention_setting.analysis == 'deepCrispr':
                bs = extra_input_for_FF.size(0)
                extra_input_for_FF = extra_input_for_FF.view(bs, -1, 4)
                d_output = cat((d_output, extra_input_for_FF), dim = 2)
            d_output = torch.unsqueeze(d_output, 1)
            d_output = self.cnn(d_output)
        flat_d_output = d_output.view(-1, d_output.size(-2)*d_output.size(-1))
        if attention_setting.add_parallel_cnn:
            src = torch.unsqueeze(src, 1)
            inter_output = self.cnn(src).view(src.size(0), -1)
            flat_d_output = cat((inter_output, flat_d_output),dim=1)
        if extra_input_for_FF is not None and attention_setting.analysis != 'deepCrispr':
            flat_d_output = cat((flat_d_output, extra_input_for_FF), dim=1)
        output = self.out(flat_d_output)
        return output

class customized_CNN(nn.Module):

    def __init__(self):

        super().__init__()
        self.cnn_1 = nn.Conv2d(1, 32, kernel_size=(3,1), padding=(1,0))
        self.maxpool_1 = nn.MaxPool2d(kernel_size=(2,1), padding=(1,0))
        self.cnn_2 = nn.Conv2d(32, 64, kernel_size=(3,1), padding=(1,0))
        if config.seq_len == 22:
            self.maxpool_2 = nn.MaxPool2d(kernel_size=(2,1), padding=(1,0))
        else:
            self.maxpool_2 = nn.MaxPool2d(kernel_size=(2,1), padding=(1,0))
        self.dropout = nn.Dropout(p = attention_setting.cnn_dropout)

    def forward(self, input):

        x = self.maxpool_1(self.cnn_1(input))
        x = F.relu(x)
        x = self.maxpool_2(self.cnn_2(x))
        x = F.relu(x)
        x = x.contiguous().view(x.size(0), -1, x.size(-1) * x.size(-2))
        return x

class OTembeddingTransformer(nn.Module):

    def __init__(self, embedding_vec_dim, d_model, N, heads, dropout, feature_len_map, classifier=False):
        super().__init__()
        self.feature_len_map = feature_len_map
        extra_length = 0 if self.feature_len_map[-1] is None else self.feature_len_map[-1][1] - self.feature_len_map[-1][0]
        d_input = self.feature_len_map[0][1] - self.feature_len_map[0][0]
        self.transformer = Transformer(d_input, d_model, embedding_vec_dim, N, heads, dropout, extra_length)
        self.embedding = nn.Embedding(config.embedding_voca_size, embedding_vec_dim)
        self.trg_embedding = nn.Embedding(config.embedding_voca_size, embedding_vec_dim)
        self.embedding_pos = nn.Embedding(d_input, embedding_vec_dim)
        self.trg_embedding_pos = nn.Embedding(d_input, embedding_vec_dim)
        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = classifier


    def forward(self, input, src_mask=None, trg_mask=None):

        src = input[:,self.feature_len_map[0][0]: self.feature_len_map[0][1]].long()
        embedded_src = self.embedding(src)
        bs = src.size(0)
        pos_len = src.size(1)
        pos = torch.from_numpy(np.array([[i for i in range(pos_len)] for _ in range(bs)]))
        pos = pos.to(OT_crispr_attn.device2)
        embedded_pos = self.embedding_pos(pos)
        embedded_src = embedded_pos + embedded_src

        if self.feature_len_map[1] is not None:
            trg = input[:, self.feature_len_map[1][0]:self.feature_len_map[1][1]].long()
        else:
            trg = src
        embedded_trg = self.trg_embedding(trg)
        embedded_pos_trg = self.trg_embedding_pos(pos)
        embedded_trg = embedded_pos_trg + embedded_trg
        embedded_src = self.dropout(embedded_src)
        embedded_trg = self.dropout(embedded_trg)
        extra_input_for_FF = None
        if self.feature_len_map[2] is not None:
            extra_input_for_FF = input[:, self.feature_len_map[2][0]: self.feature_len_map[2][1]]
        output = self.transformer(embedded_src, embedded_trg, extra_input_for_FF=extra_input_for_FF,
                                  src_mask=src_mask, trg_mask=trg_mask)
        if self.classifier:
            # output = F.log_softmax(output, dim = -1)
            #output = F.softmax(output, dim = -1)
            pass
        return output


class EmbeddingTransformer(Transformer):

    def __init__(self, embedding_vec_dim , d_input, d_model, N, heads, dropout, extra_length):
        super().__init__(d_input, d_model, embedding_vec_dim, N, heads, dropout, extra_length)
        self.embedding = nn.Embedding(config.embedding_voca_size, embedding_vec_dim)
        self.embedding_2 = nn.Embedding(config.embedding_voca_size, embedding_vec_dim)
        self.trg_embedding = nn.Embedding(config.embedding_voca_size, embedding_vec_dim)
        self.embedding_pos = nn.Embedding(config.seq_len - config.word_len + 1, embedding_vec_dim)
        self.trg_embedding_pos = nn.Embedding(config.seq_len - config.word_len + 1, embedding_vec_dim)
        self.dropout = nn.Dropout(p = config.dropout)

    def forward(self, src, trg = None, extra_input_for_FF=None, src_mask=None, trg_mask=None):

        if config.sep_len != 0:
            src_1 = src[:,:config.sep_len]
            src_2 = src[:, config.sep_len:]
            embedded_src = self.embedding(src_1)
            embedded_src_2 = self.embedding_2(src_2)
            embedded_src = cat(tuple([embedded_src, embedded_src_2]), dim=1)
        else:
            embedded_src = self.embedding(src)

        bs = src.size(0)
        pos_length = config.seq_len - config.seq_start - config.word_len + 1
        pos = torch.from_numpy(np.array([[i for i in range(pos_length)] for _ in range(bs)]))
        pos = pos.to(crispr_attn.device2)
        embedded_src_pos = self.embedding_pos(pos)
        embedded_src_1 = embedded_src + embedded_src_pos
        embedded_src_2 = self.dropout(embedded_src_1)

        if trg is not None:
            embedded_trg = self.trg_embedding(trg)
            embedded_trg_pos = self.trg_embedding_pos(pos)
            embedded_trg_1 = embedded_trg + embedded_trg_pos
            embedded_trg_2 = self.dropout(embedded_trg_1)
        else:
            embedded_trg_2 = embedded_src_2

        #embedded_src_2 = transpose(embedded_src_2, 1, 2)
        output = super().forward(embedded_src_2, embedded_trg_2, extra_input_for_FF)
        return output

def get_OT_model(feature_len_map, classifier = False):

    assert attention_setting.d_model % attention_setting.attention_heads == 0
    assert attention_setting.attention_dropout < 1
    if not classifier:
        model = OTembeddingTransformer(attention_setting.n_feature_dim, attention_setting.d_model,
                                   attention_setting.n_layers, attention_setting.attention_heads,
                                   attention_setting.attention_dropout, feature_len_map)
    else:
        model = OTembeddingTransformer(attention_setting.n_feature_dim, attention_setting.d_model,
                                       attention_setting.n_layers, attention_setting.attention_heads,
                                       attention_setting.attention_dropout, feature_len_map, classifier = True)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


def get_model(inputs_lengths=None, d_input = 20):
    assert attention_setting.d_model % attention_setting.attention_heads == 0
    assert attention_setting.attention_dropout < 1

    #model = Transformer(d_input, attention_setting.d_model, attention_setting.n_feature_dim, attention_setting.n_layers, attention_setting.attention_heads, attention_setting.attention_dropout)

    extra_feature_length = len(config.extra_categorical_features + config.extra_numerical_features)
    model = EmbeddingTransformer(attention_setting.n_feature_dim, d_input, attention_setting.d_model,
                                 attention_setting.n_layers, attention_setting.attention_heads,
                                 attention_setting.attention_dropout, extra_feature_length)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    # if attention_setting.device == 0:
    #     model = model.cuda()

    return model
