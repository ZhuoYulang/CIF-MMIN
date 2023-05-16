import torch
from models.networks.LightWeightTrans import TransEncoder


class DoubleTrans(torch.nn.Module):
    def __init__(self, main_dim, middle_dim, d_model, num_head,
                 num_layer, dim_forward, p=0.5):    # 2023/2/8: delete device and config which is a parameter
        super().__init__()
        # self.lr = config.lr
        self.use_reconst_loss = True
        self.g12 = TransEncoder(d_dual=(main_dim, middle_dim), d_model=d_model[1], nhead=num_head[1],
                                num_encoder_layers=num_layer[1],
                                dim_feedforward=dim_forward[1], dropout=p)

        self.g21 = TransEncoder(d_dual=(middle_dim, main_dim), d_model=d_model[0], nhead=num_head[0],
                                num_encoder_layers=num_layer[0],
                                dim_feedforward=dim_forward[0], dropout=p)

        # self.g12_optimizer = torch.optim.Adam(self.g12.parameters(), config.a2t_lr, (config.beta1, config.beta2))
        # self.g21_optimizer = torch.optim.Adam(self.g21.parameters(), config.t2a_lr, (config.beta1, config.beta2))

        # self.g12.to(device)
        # self.g21.to(device)

    def reset_grad(self):
        # self.g_optimizer.zero_grad()
        self.g12_optimizer.zero_grad()
        self.g21_optimizer.zero_grad()

    def grad_step(self):
        self.g12_optimizer.step()
        self.g21_optimizer.step()

    def double_fusion(self, source, target, need_grad=False):
        # self.reset_grad()
        if need_grad:
            fake_target, bimodal_12 = self.g12(source)
            fake_source, bimodal_21 = self.g21(target)
        else:
            self.g12.eval()
            self.g21.eval()
            with torch.no_grad():
                fake_target, bimodal_12 = self.g12(source)
                fake_source, bimodal_21 = self.g21(target)
        return fake_source, fake_target, bimodal_12, bimodal_21

    # def train(self, source, target):
    def forward(self, source, target):
        # self.g12.train()
        # self.g21.train()

        # train with source-target-source cycle
        # self.reset_grad()
        fake_target, bimodal_12 = self.g12(source)
        reconst_source, bimodal_21 = self.g21(fake_target)
        # g_loss = torch.mean((source - reconst_source) ** 2)
        # g_loss.backward()
        # self.grad_step()

        # train with target-source-target
        # self.reset_grad()
        fake_source, _ = self.g21(target)
        reconst_target, _ = self.g12(fake_source)
        # g_loss = torch.mean((target - reconst_target) ** 2)
        # g_loss.backward()
        # self.grad_step()
        return fake_target, reconst_source, fake_source, reconst_target, bimodal_12, bimodal_21