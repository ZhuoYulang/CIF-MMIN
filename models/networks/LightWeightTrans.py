import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, Dropout, Module
from torch.nn.modules import ModuleList
from models.networks.multihead_attention import MultiheadAttention


def _get_clones(module, n):
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


class TransEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm = LayerNorm(d_model)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm(src)
        return src


class TransEncoder(Module):
    def __init__(self, d_dual, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransEncoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_encoder_layers
        self.linear1 = Linear(d_dual[0], d_model)
        self.linear2 = Linear(d_model, d_dual[1])
        self.dropout = Dropout(dropout)

        encoder_layer = TransEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

        self.norm = LayerNorm(d_model)

    def forward(self, src, mask=None, src_key_padding_mask=None):
        res = list()
        output = self.dropout(F.relu(self.linear1(src)))
        res.append(output)
        for i in range(self.num_layers):
            output = self.layers[i](output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            res.append(output)
        if self.norm:
            output = self.norm(output)
            res.append(output)
        return self.linear2(output), res


class EmotionClassifier(nn.Module):
    def __init__(self, config):
        super(EmotionClassifier, self).__init__()
        self.gpu_ids = config.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')

        self.output_dim = config.output_dim
        self.rnn_dropout = nn.Dropout(p=0.3, inplace=True)
        self.rnn_text = nn.LSTM(input_size=config.input_dim_l, hidden_size=config.gru_units,  # text_dim->input_dim_l
                                num_layers=1, bidirectional=False, dropout=0.0, batch_first=True)
        self.rnn_audio = nn.LSTM(input_size=config.input_dim_a, hidden_size=config.gru_units,  # audio_dim->input_dim_a
                                 num_layers=1, bidirectional=False, dropout=0.0, batch_first=True)

        self.dense_text = nn.Linear(in_features=config.gru_units * 1, out_features=config.dense_units)
        self.dense_audio = nn.Linear(in_features=config.gru_units * 1, out_features=config.dense_units)
        self.dense_dropout = nn.Dropout(p=0.3, inplace=True)

        cat_dims = config.a_d_model + config.t_d_model + config.dense_units * 2
        self.out_layer_1 = nn.Linear(in_features=cat_dims, out_features=config.dense_units)
        self.out_layer_2 = nn.Linear(in_features=config.dense_units, out_features=config.output_dim)
        self.out_dropout = nn.Dropout(p=0.3, inplace=True)

    def forward(self, audio, text, uni_fusion):
        rnn_t, _ = self.rnn_text(text)
        encoded_text = torch.relu(self.dense_dropout(self.dense_text(torch.relu(rnn_t))))
        rnn_a, _ = self.rnn_audio(audio)
        encoded_audio = torch.relu(self.dense_dropout(self.dense_audio(torch.relu(rnn_a))))

        encoded_text = encoded_text.view(encoded_text.size(0), encoded_text.size(-1), encoded_text.size(1))
        encoded_audio = encoded_audio.view(encoded_audio.size(0), encoded_audio.size(-1), encoded_audio.size(1))

        layer_3 = torch.nn.Linear(in_features=encoded_text.size(-1), out_features=64).to(self.device)
        encoded_text = layer_3(encoded_text)
        layer_4 = torch.nn.Linear(in_features=encoded_audio.size(-1), out_features=64).to(self.device)
        encoded_audio = layer_4(encoded_audio)

        encoded_text = encoded_text.view(encoded_text.size(0), encoded_text.size(-1), encoded_text.size(1))
        encoded_audio = encoded_audio.view(encoded_audio.size(0), encoded_audio.size(-1), encoded_audio.size(1))

        encoded_feature = torch.cat((encoded_text, encoded_audio, uni_fusion[0], uni_fusion[1]), dim=-1)
        out1 = self.out_dropout(torch.relu(self.out_layer_1(encoded_feature)))
        out2 = self.out_layer_2(out1)
        in_feat = out2.transpose(1, 2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)
        # return self.out_layer_2(out1)  # mode = sentiment


'''
D1 = Discriminator(feature_dims=128, conv_dim=20)
dat = torch.randn(13, 1, 20, 128)
output = D1(dat)
print(output.shape)
torch.Size([13, 20, 10, 64])
torch.Size([13, 40, 5, 32])
torch.Size([13, 80, 2, 16])
torch.Size([13, 1, 1, 1])
Translation = TransEncoder(d_dual=(300, 5), d_model=128, nhead=4, num_encoder_layers=2,
                           dim_feedforward=512, dropout=0.5)
dat = torch.empty(20, 11, 300)
res = Translation(dat)
print(res.shape)
print(res[0, 1])
'''
