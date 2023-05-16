import torch
import os
import json
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.networks.classifier import FcClassifier, Fusion
from models.networks.shared import SharedEncoder
from models.utils import CMD
from einops import rearrange, repeat, reduce
from torch import einsum


def masked_mean(t, mask, dim=1, eps=1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim=dim)
    denom = mask.sum(dim=dim).clamp(min=eps)
    return numer / denom


def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device=device)
    j_range = torch.arange(j, device=device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d=num_diag_el)


def log(t, eps=1e-20):
    return torch.log(t + eps)


def l2norm(t):
    return F.normalize(t, dim=-1)


class UttSelfSuperviseModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='visual input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='attention', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
                            help='visual embedding method,last,mean or atten')
        parser.add_argument('--cls_layers', type=str, default='128,128',
                            help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--modality', type=str, help='which modality to use for model')
        parser.add_argument('--image_dir', type=str, default='./consistent_image', help='models image are saved here')
        return parser

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['TA', 'TV', 'VA', 'CE']
        self.modality = opt.modality
        self.model_names = ['SharedA', 'SharedV', 'SharedT', "C"]
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        cls_input_size = opt.embd_size_a * int("A" in self.modality) + \
                         opt.embd_size_v * int("V" in self.modality) + \
                         opt.embd_size_l * int("L" in self.modality)
        self.netSharedV = SharedEncoder(opt)
        self.netSharedA = SharedEncoder(opt)
        self.netSharedT = SharedEncoder(opt)
        if self.opt.corpus_name != 'MOSI':
            self.netC = FcClassifier(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate,
                                     use_bn=opt.bn)
        else:
            self.netC = Fusion(cls_input_size, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate)

        self.temperature = torch.nn.Parameter(torch.tensor(1.))
        self.batch_size = opt.batch_size
        # 视频和音频是使用RNN，文本使用TextCNN
        # acoustic model
        if 'A' in self.modality:
            self.model_names.append('A')
            self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
            self.model_names.append('ConA')
            self.netConA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)

        # lexical model
        if 'L' in self.modality:
            self.model_names.append('L')
            self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
            self.model_names.append('ConL')
            self.netConL = LSTMEncoder(opt.input_dim_l, opt.embd_size_l, embd_method=opt.embd_method_a)

        # visual model
        if 'V' in self.modality:
            self.model_names.append('V')
            self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)
            self.model_names.append('ConV')
            self.netConV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)

        if self.isTrain:
            if self.opt.corpus_name != 'MOSI':
                self.criterion_ce = torch.nn.CrossEntropyLoss()
            else:
                self.criterion_ce = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net' + net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999),
                                              weight_decay=opt.weight_decay)
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        # # feature save_dir
        # image_save_dir = os.path.join(opt.image_dir, opt.name)
        # image_save_dir = os.path.join(image_save_dir, str(opt.cvNo))
        # self.consistent_image_save_dir = os.path.join(image_save_dir, 'consistent')
        # # print(self.consistent_image_save_dir)
        # if not os.path.exists(self.consistent_image_save_dir):
        #     os.makedirs(self.consistent_image_save_dir)

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        if 'A' in self.modality:
            self.acoustic = input['A_feat'].float().to(self.device)
        if 'L' in self.modality:
            self.lexical = input['L_feat'].float().to(self.device)
        if 'V' in self.modality:
            self.visual = input['V_feat'].float().to(self.device)

        self.label = input['label'].to(self.device)

        if self.opt.corpus_name == 'MOSI':
            self.label = self.label.unsqueeze(1)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        final_embd = []
        final_shared = []
        if 'A' in self.modality:
            self.feat_A = self.netA(self.acoustic)
            final_embd.append(self.feat_A)
            feat_ConA = self.netConA(self.acoustic)
            self.feat_shared_A = self.netSharedA(feat_ConA)
            final_shared.append(feat_ConA)

        if 'L' in self.modality:
            self.feat_L = self.netL(self.lexical)  # netL：LSTM
            final_embd.append(self.feat_L)
            feat_ConL = self.netConL(self.lexical)
            self.feat_shared_T = self.netSharedT(feat_ConL)
            final_shared.append(feat_ConL)

        if 'V' in self.modality:
            self.feat_V = self.netV(self.visual)  # netV：LSTM
            final_embd.append(self.feat_V)
            feat_ConV = self.netConV(self.visual)
            self.feat_shared_V = self.netSharedV(feat_ConV)
            final_shared.append(feat_ConV)

        # get model outputs
        self.feat = torch.cat(final_embd, dim=-1)

        self.logits, self.ef_fusion_feat = self.netC(self.feat)
        if self.opt.corpus_name != "MOSI":
            self.pred = F.softmax(self.logits, dim=-1)
        else:
            self.pred = self.logits

    def backward(self):
        """Calculate the loss for back propagation"""

        num_batch_texts = 1
        # get temperature

        temp = self.temperature.exp()
        temperature = torch.nn.Parameter(torch.tensor(1e-10))
        self.feat_shared_V = rearrange(self.feat_shared_V, '(m b) ... -> m b ...', m=num_batch_texts)
        self.feat_shared_T = rearrange(self.feat_shared_T, '(m b) ... -> m b ...', m=num_batch_texts)
        self.feat_shared_A = rearrange(self.feat_shared_A, '(m b) ... -> m b ...', m=num_batch_texts)
        # print('utt_self-first_feat_shared_V is:', self.feat_shared_V)

        self.feat_shared_T, self.feat_shared_V = map(l2norm, (self.feat_shared_T, self.feat_shared_V))
        self.feat_shared_T, self.feat_shared_A = map(l2norm, (self.feat_shared_T, self.feat_shared_A))
        self.feat_shared_V, self.feat_shared_A = map(l2norm, (self.feat_shared_V, self.feat_shared_A))
        # print('utt_self-second_feat_shared_V is:', self.feat_shared_V)

        self.text_to_image = einsum('m t d, n i d -> m n t i', self.feat_shared_T, self.feat_shared_V) * temp
        self.image_to_text = rearrange(self.text_to_image, '... t i -> ... i t')
        # print('utt_self-first_text_to_image is:', self.text_to_image)

        self.text_to_audio = einsum('m t d, n i d -> m n t i', self.feat_shared_T, self.feat_shared_A) * temp
        self.audio_to_text = rearrange(self.text_to_audio, '... t i -> ... i t')

        self.image_to_audio = einsum('m t d, n i d -> m n t i', self.feat_shared_V, self.feat_shared_A) * temp
        self.audio_to_image = rearrange(self.image_to_audio, '... t i -> ... i t')

        # calculate loss
        self.text_to_image = rearrange(self.text_to_image, 'm n ... -> (m n) ...')
        self.image_to_text = rearrange(self.image_to_text, 'm n ... -> (m n) ...')
        # print('utt_self-second_text_to_image is:', self.text_to_image)

        self.text_to_audio = rearrange(self.text_to_audio, 'm n ... -> (m n) ...')
        self.audio_to_text = rearrange(self.audio_to_text, 'm n ... -> (m n) ...')

        self.image_to_audio = rearrange(self.image_to_audio, 'm n ... -> (m n) ...')
        self.audio_to_image = rearrange(self.audio_to_image, 'm n ... -> (m n) ...')

        # exponentiate

        text_to_image_exp, image_to_text_exp = map(torch.exp, (self.text_to_image, self.image_to_text))
        text_to_audio_exp, audio_to_text_exp = map(torch.exp, (self.text_to_audio, self.audio_to_text))
        image_to_audio_exp, audio_to_image_exp = map(torch.exp, (self.image_to_audio, self.audio_to_image))

        # numerators

        text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))
        text_to_audio_pos, audio_to_text_pos = map(matrix_diag, (text_to_audio_exp, audio_to_text_exp))
        image_to_audio_pos, audio_to_image_pos = map(matrix_diag, (image_to_audio_exp, audio_to_image_exp))

        # denominator

        pos_mask = torch.eye(self.lexical.shape[0], device=self.device, dtype=torch.bool)
        text_to_image_exp, image_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.),
                                                   (text_to_image_exp, image_to_text_exp))
        text_to_audio_exp, audio_to_text_exp = map(lambda t: t.masked_fill(pos_mask, 0.),
                                                   (text_to_audio_exp, audio_to_text_exp))
        image_to_audio_exp, audio_to_image_exp = map(lambda t: t.masked_fill(pos_mask, 0.),
                                                     (image_to_audio_exp, audio_to_image_exp))

        text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim=-1), (text_to_image_exp, image_to_text_exp))
        text_to_audio_denom, audio_to_text_denom = map(lambda t: t.sum(dim=-1), (text_to_audio_exp, audio_to_text_exp))
        image_to_audio_denom, audio_to_image_denom = map(lambda t: t.sum(dim=-1),
                                                         (image_to_audio_exp, audio_to_image_exp))

        text_to_image_loss = -log(text_to_image_pos / (text_to_image_denom + temperature)).mean(dim=-1)
        image_to_text_loss = -log(image_to_text_pos / (image_to_text_denom + temperature)).mean(dim=-1)

        text_to_audio_loss = -log(text_to_audio_pos / (text_to_audio_denom + temperature)).mean(dim=-1)
        audio_to_text_loss = -log(audio_to_text_pos / (audio_to_text_denom + temperature)).mean(dim=-1)

        # print('audio_to_image_pos is:', audio_to_image_pos)
        # print('audio_to_image_denom is:', audio_to_image_denom + temperature)
        image_to_audio_loss = -log(image_to_audio_pos / (image_to_audio_denom + temperature)).mean(dim=-1)
        audio_to_image_loss = -log(audio_to_image_pos / (audio_to_image_denom + temperature)).mean(dim=-1)

        cl_tv_losses = (text_to_image_loss + image_to_text_loss) / 2
        cl_ta_losses = (text_to_audio_loss + audio_to_text_loss) / 2
        cl_va_losses = (image_to_audio_loss + audio_to_image_loss) / 2

        cl_tv_loss = cl_tv_losses[0]
        cl_ta_loss = cl_ta_losses[0]
        cl_va_loss = cl_va_losses[0]

        tv_weight = 0.25
        ta_weight = 0.25
        va_weight = 0.5
        self.loss_TA = (cl_ta_loss * ta_weight)
        self.loss_TV = (cl_tv_loss * tv_weight)
        self.loss_VA = (cl_va_loss * va_weight)
        if self.opt.corpus_name != 'MOSI':
            self.loss_CE = self.criterion_ce(self.logits, self.label)
        else:
            self.loss_CE = self.criterion_ce(self.logits, self.label)
        loss = self.loss_CE + self.loss_TV + self.loss_TA + self.loss_VA
        # torch.autograd.set_detect_anomaly(True)
        # with torch.autograd.detect_anomaly():
        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net' + model).parameters(), 0.5)

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()
        # backward
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
