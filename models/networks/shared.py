import torch
import torch.nn as nn

class SharedEncoder(nn.Module):
    def __init__(self, opt):
        ''' Fully Connect classifier
            fc+relu+bn+dropout， 最后分类128-4层是直接fc的
            Parameters:
            --------------------------
            input_dim: input feature dim
            layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
            dropout: dropout rate
            use_bn: use batchnorm or not
        '''
        super().__init__()
        input_dim = opt.embd_size_a
        self.netshared = nn.Sequential()  # 一个全连接层，加一个Sigmod层，用于激活
        # print(type(opt.run_idx))

        self.netshared.add_module('shared_1', nn.Linear(in_features=input_dim, out_features=input_dim))
        self.netshared.add_module('shared_1_activation', nn.LeakyReLU())
        self.netshared.add_module('layer_norm', nn.LayerNorm(input_dim))
        self.netshared.add_module('shared_2', nn.Linear(in_features=input_dim, out_features=input_dim))
        self.netshared.add_module('shared_2_activation', nn.LeakyReLU())
    
    def forward(self, x):
        ## make layers to a whole module
        feat = self.netshared(x)
        return feat