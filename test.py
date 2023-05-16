import os
import json
from typing import List
import torch
import numpy as np
import h5py
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
import pickle


if __name__ == '__main__':
    pwd = os.path.abspath(__file__)
    pwd = os.path.dirname(pwd)
    # config = json.load(open(os.path.join(pwd, 'data', 'config', f'IEMOCAP_config.json')))
    # all_L = \
    #     h5py.File(os.path.join(pwd, 'data', 'IEMOCAP_features_2021', 'L', f'bert_large.h5'), 'r')
    # print(all_L.items())

    f = open(os.path.join(pwd, 'data', 'mosi_MISA', f'dev.pkl'), 'rb')
    # f = open(os.path.join(pwd, 'data', 'mosi_features', 'raw', f'audio_2way.pickle'), 'rb')
    data = pickle.load(f)
    print(data[0][0][0].shape)