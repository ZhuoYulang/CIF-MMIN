import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    ''' one directional LSTM encoder
    '''
    def __init__(self, input_size, hidden_size, embd_method='last', bidirectional=False):
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, bidirectional=bidirectional)
        assert embd_method in ['maxpool', 'attention', 'last', 'dense']
        self.embd_method = embd_method
        
        if self.embd_method == 'attention':
            self.attention_vector_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
            self.attention_layer = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.softmax = nn.Softmax(dim=-1)
        elif self.embd_method == 'dense':
            self.dense_layer = nn.Sequential()
            self.bidirectional = bidirectional
            if bidirectional:
                self.dense_layer.add_module('linear', nn.Linear(2 * self.hidden_size, self.hidden_size))
            else:
                self.dense_layer.add_module('linear', nn.Linear(self.hidden_size, self.hidden_size))
            self.dense_layer.add_module('activate', nn.Tanh())
            self.softmax = nn.Softmax(dim=-1)

    def embd_attention(self, r_out, h_n):
        ''''
        参考这篇博客的实现:
        https://blog.csdn.net/dendi_hust/article/details/94435919
        https://blog.csdn.net/fkyyly/article/details/82501126
        论文: Hierarchical Attention Networks for Document Classification
        formulation:  lstm_output*softmax(u * tanh(W*lstm_output + Bias)
        W and Bias 是映射函数，其中 Bias 可加可不加
        u 是 attention vector 大小等于 hidden size
        '''
        hidden_reps = self.attention_layer(r_out)                       # [batch_size, seq_len, hidden_size]
        atten_weight = (hidden_reps @ self.attention_vector_weight)              # [batch_size, seq_len, 1]
        atten_weight = self.softmax(atten_weight)                       # [batch_size, seq_len, 1]
        # [batch_size, seq_len, hidden_size] * [batch_size, seq_len, 1]  =  [batch_size, seq_len, hidden_size]
        sentence_vector = torch.sum(r_out * atten_weight, dim=1)       # [batch_size, hidden_size]
        return sentence_vector

    def embd_maxpool(self, r_out, h_n):
        # embd = self.maxpool(r_out.transpose(1,2))   # r_out.size()=>[batch_size, seq_len, hidden_size]
                                                    # r_out.transpose(1, 2) => [batch_size, hidden_size, seq_len]
        in_feat = r_out.transpose(1,2)
        embd = F.max_pool1d(in_feat, in_feat.size(2), in_feat.size(2))
        return embd.squeeze(-1)

    def embd_last(self, r_out, h_n):
        #Just for  one layer and single direction
        return h_n.squeeze(0)

    def embd_dense(self, r_out, h_n):
        if self.bidirectional:
            h_n = torch.cat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)
            output = self.dense_layer(h_n)
        else:
            h_n = h_n[-1, :, :]
            output = self.dense_layer(h_n)
        return output

    def forward(self, x):
        '''
        r_out shape: seq_len, batch, num_directions * hidden_size
        hn and hc shape: num_layers * num_directions, batch, hidden_size
        '''
        r_out, (h_n, h_c) = self.rnn(x)
        embd = getattr(self, 'embd_'+self.embd_method)(r_out, h_n)
        return embd
        # return r_out