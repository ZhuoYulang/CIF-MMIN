import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.AE_input_dim = opt.embd_size_a + opt.embd_size_v + opt.embd_size_l
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.AE_input_dim, nhead=opt.nhead, dropout=opt.encoder_dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=opt.n_blocks)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.AE_input_dim, nhead=opt.nhead, dropout=opt.decoder_dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=opt.n_blocks)

    def forward(self, x, fusion_feat_miss):
        latent = self.encoder(x)
        out = self.decoder(fusion_feat_miss, latent)
        return out, latent