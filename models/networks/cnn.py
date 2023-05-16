import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, input_dim, emb_size=128, in_channels=1, out_channels=128, kernel_size=5, dropout=0.5):
        super().__init__()
        '''
        cat((conv1-relu+conv2-relu+conv3-relu)+maxpool) + dropout, and to trans
        '''
        # self.conv1 = nn.Conv2d(in_channels, out_channels, (kernel_heights[0], input_dim), stride=1, padding=0)
        # self.conv2 = nn.Conv2d(in_channels, out_channels, (kernel_heights[1], input_dim), stride=1, padding=0)
        # self.conv3 = nn.Conv2d(in_channels, out_channels, (kernel_heights[2], input_dim), stride=1, padding=0)
        # self.dropout = nn.Dropout(dropout)
        # self.embd = nn.Sequential(
        #     nn.Linear(len(kernel_heights)*out_channels, emb_size),
        #     nn.ReLU(inplace=True),
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (kernel_size, input_dim), stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=1)
        )

    def conv_block(self, input, conv_layer):
        conv_out = conv_layer(input)# conv_out.size() = (batch_size, out_channels, dim, 1)
        activation = F.relu(conv_out.squeeze(3))# activation.size() = (batch_size, out_channels, dim1)
        max_out = F.max_pool1d(activation, activation.size()[2]).squeeze(2) # maxpool_out.size() = (batch_size, out_channels)
        return max_out

    def forward(self, frame_x):
        # batch_size, seq_len, feat_dim = frame_x.size()
        # frame_x = frame_x.view(batch_size, 1, seq_len, feat_dim)
        # max_out1 = self.conv_block(frame_x, self.conv1)
        # max_out2 = self.conv_block(frame_x, self.conv2)
        # max_out3 = self.conv_block(frame_x, self.conv3)
        # all_out = torch.cat((max_out1, max_out2, max_out3), 1)
        # fc_in = self.dropout(all_out)
        # embd = self.embd(fc_in)
        out = self.conv1(frame_x)  # embd.shape: [batch_size, out_channels, dim, 1]
        embd = out.view(frame_x.size(0), -1)
        return embd