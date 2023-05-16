import torch
import torch.nn as nn

class ProjectEncoder(nn.Module):
    def __init__(self, in_features, out_features, activate=torch.nn.ReLU()):
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
        self.netproject = nn.Sequential()  # 一个全连接层，加一个Sigmod层，用于激活
        self.netproject.add_module('project_v', nn.Linear(in_features=in_features, out_features=out_features))
        self.netproject.add_module('project_v_activation', activate)
        self.netproject.add_module('project_v_layer_norm', nn.LayerNorm(out_features))

    
    def forward(self, x):
        feat = self.netproject(x)
        return feat