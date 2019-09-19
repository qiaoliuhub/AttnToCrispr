import torch.nn.functional as F
import torch.nn as nn
from Sublayers import FeedForward, MultiHeadAttention, Norm
import attention_setting

class EncoderLayer(nn.Module):
    def __init__(self, d_input, d_model, heads, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(d_input, d_model)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):

        x = F.relu(self.input_linear(x))
        x2 = self.norm_1(x) if attention_setting.attention_layer_norm else x
        x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        x2 = self.norm_2(x) if attention_setting.attention_layer_norm else x
        x = x + self.dropout_2(self.ff(x2))
        return x


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):
    def __init__(self, d_input, d_model, heads, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(d_input, d_model)
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_outputs, src_mask=None, trg_mask=None):

        x = F.relu(self.input_linear(x))
        x2 = self.norm_1(x) if attention_setting.attention_layer_norm else x
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, trg_mask))
        x2 = self.norm_2(x) if attention_setting.attention_layer_norm else x
        x = x + self.dropout_2(self.attn_2(x2, e_outputs, e_outputs, src_mask))
        x2 = self.norm_3(x) if attention_setting.attention_layer_norm else x
        x = x + self.dropout_3(self.ff(x2))
        return x