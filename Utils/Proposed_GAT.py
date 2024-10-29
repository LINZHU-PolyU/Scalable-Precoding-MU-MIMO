import torch
import torch.nn as nn
import torch.nn.functional as F
from Utils.residual_vq import RVQ


class GNN_init_layer(nn.Module):
    def __init__(self, vq_dim):
        super(GNN_init_layer, self).__init__()
        self.vq_dim = vq_dim

        self.linear1 = nn.Sequential(
            nn.Linear(in_features=self.vq_dim,
                      out_features=1024),
            nn.Mish(),
        )

        self.linear2 = nn.Sequential(
            nn.Linear(in_features=1024,
                      out_features=512),
            nn.Mish(),
        )

    def forward(self, inputs):
        x = self.linear1(inputs)
        x = self.linear2(x)

        return x


class GNN_layer(nn.Module):
    def __init__(self, no_ue):
        super(GNN_layer, self).__init__()
        self.no_ue = no_ue

        self.MLPBlock1 = nn.Linear(512, 512)
        self.MLPBlock2 = nn.Linear(512, 512)
        self.MLPBlock3 = nn.Linear(512, 512)
        self.MLPBlock4 = nn.Linear(512, 512)

        self.Mish = nn.Mish()

    def forward(self, x):
        # x: Output from previous layer --> x = [x_1, ..., x_K] --> x_i: b x 512
        x_out = []  # x_out = [x_out_1, ..., x_out_K] --> x_out_i: b x 512
        for i in range(self.no_ue):
            MLP1_res_i = self.MLPBlock1(x[i])  # MLP1_res_i: b x 512
            MLP4_res_i = self.MLPBlock4(x[i])  # MLP4_res_i: b x 512
            tmp_j_list = []
            for j in range(self.no_ue):
                if i != j:
                    MLP2_res_j = self.MLPBlock2(x[j])  # MLP2_res_j: b x 512
                    MLP3_res_j = self.MLPBlock3(x[j])  # MLP3_res_j: b x 512

                    Beta_jk = MLP4_res_i * MLP3_res_j  # Beta: b x 512

                    tmp_j = MLP2_res_j * Beta_jk  # tmp_j: b x 512
                    tmp_j_list.append(tmp_j)  # tmp_j_list = [tmp_1, ..., tmp_{j-1}, tmp_{j+1}, ..., tmp_K]  -->
                                              # tmp_j: b x 512

            tmp_stack = torch.stack(tmp_j_list, dim=-1)  # tmp_stack: b x 512 x (K-1)
            tmp_mean = torch.mean(tmp_stack, dim=-1)  # tmp_mean: b x 512
            x_out_i = self.Mish(MLP1_res_i + tmp_mean)  # x_out_i: b x 512

            x_out.append(x_out_i)

        return x_out


class Pilot_Network(nn.Module):
    def __init__(self, BS_antenna, UE_antenna, time_samples):
        super(Pilot_Network, self).__init__()
        self.BS_antenna = BS_antenna  # M
        self.UE_antenna = UE_antenna  # K
        self.time_samples = time_samples  # L

        W_real = torch.empty(self.BS_antenna, self.time_samples)  # M x L
        W_imag = torch.empty(self.BS_antenna, self.time_samples)  # M x L

        # Initialize W_real and W_imag using Xavier
        nn.init.xavier_uniform_(W_real)
        nn.init.xavier_uniform_(W_imag)

        self.W_real = nn.Parameter(W_real, requires_grad=True)  # M x L
        self.W_imag = nn.Parameter(W_imag, requires_grad=True)  # M x L

    def forward(self, h):  # Input channel
        # h -> complex, size = b x K x M

        # Scale and Reshape BS precoding
        W = self.W_real + 1j * self.W_imag  # W: M x L
        W_norm = (torch.linalg.vector_norm(W, dim=0))
        W = W / W_norm.unsqueeze(0)  # W: M x L

        # Get equivalent pilots
        pilot = h @ W.unsqueeze(0)  # pilot: b x K x L
        pilot_real = pilot.real  # pilot_real: b x K x L
        pilot_imag = pilot.imag  # pilot_imag: b x K x L

        return pilot_real, pilot_imag


class encoder(nn.Module):
    def __init__(self, x_dim, vq_dim):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 1024),  # x_dim: dimension of input = 2L
            nn.BatchNorm1d(1024),
            nn.Mish(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Mish(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Mish(),
            nn.Linear(256, vq_dim),  # vq_dim: dimension of extracted feature
            nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)


class myModel(nn.Module):
    def __init__(self, vq_dim, vq_b, n_ue, n_quantizer, BS_ant, UE_ant, time_samples, SNR):
        super(myModel, self).__init__()
        self.SNR = SNR
        self.n_ue = n_ue  # K
        self.BS_ant = BS_ant  # M
        self.UE_ant = UE_ant
        x_dim = 2 * time_samples  # 2L
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.pilot = Pilot_Network(BS_ant, UE_ant, time_samples)  # Pilot Network

        self.encoder = encoder(x_dim, vq_dim)  # Common Feature Extractor for all UEs

        self.vq = RVQ(n_quantizer, vq_b, vq_dim, device=device)  # Common VQ for all UEs

        # GAT Precoding
        self.GNN_init_layer = GNN_init_layer(vq_dim)
        self.GNN_layer_1 = GNN_layer(n_ue)
        self.GNN_layer_2 = GNN_layer(n_ue)
        self.MLP_out = nn.Sequential(
            nn.Linear(512, 256),
            nn.Mish(),
            nn.Linear(256, 2 * BS_ant),
        )

    def forward(self, x, train_mode=True):
        # x -> complex, size = b x K x M

        # Get pilots
        pilot_real, pilot_imag = self.pilot(x)  # pilot_real: b x K x L, pilot_imag: b x K x L

        # Compute noise power & generate noise
        n_power = 10 ** (-self.SNR / 10)
        noise = (torch.normal(0, 1, size=pilot_real.size()) * torch.sqrt(torch.tensor([n_power / 2]))).cuda()
        pilot_real_noisy = pilot_real + noise  # pilot_real_noisy: b x K x L
        pilot_imag_noisy = pilot_imag + noise  # pilot_imag_noisy: b x K x L

        # Encoder
        encoder_res = []
        vq_res = []
        for i in range(self.n_ue):
            pilot_real_i = pilot_real_noisy[:, i, :].squeeze()  # pilot_real_i: b x L
            pilot_imag_i = pilot_imag_noisy[:, i, :].squeeze()  # pilot_imag_i: b x L
            pilot_i = torch.cat((pilot_real_i, pilot_imag_i), dim=-1)  # pilot_i: b x 2L
            encoder_i_res = self.encoder(pilot_i)  # encoder_res: b x vq_dim
            vq_i_res, _ = self.vq(encoder_i_res, train_mode)  # vq_i_res: b x vq_dim
            encoder_res.append(encoder_i_res)
            vq_res.append(vq_i_res)

        # Compute quantization loss
        encoder_res_cat = torch.cat(encoder_res, dim=-1)  # encoder_res_cat: b x (vq_dim * K)
        vq_res_cat = torch.cat(vq_res, dim=-1)  # vq_res_cat: b x (vq_dim * K)
        vq_loss = F.mse_loss(vq_res_cat, encoder_res_cat)

        # GAT Precoding
        x_w = []
        # Initialization layer
        for i in range(self.n_ue):
            vq_i_res = vq_res[i]
            GNN_init_res = self.GNN_init_layer(vq_i_res)  # GNN_init_res: b x 512
            x_w.append(GNN_init_res)  # x_w = [x_w_1, ..., x_w_K] --> x_w_i: b x 512

        # Updating layers
        x_w = self.GNN_layer_1(x_w)  # x_w = [x_w_1, ..., x_w_K] --> x_w_i: b x 512
        x_w = self.GNN_layer_2(x_w)  # x_w = [x_w_1, ..., x_w_K] --> x_w_i: b x 512

        # Reshape layer
        W = []
        for i in range(self.n_ue):
            x_w_i = x_w[i]  # x_w_i: b x 512
            x_w_i = self.MLP_out(x_w_i)  # x_w_i: b x 2M
            x_w_i_r = x_w_i[:, :self.BS_ant]  # x_w_i_r: b x M
            x_w_i_i = x_w_i[:, self.BS_ant:]
            x_w_i_C = x_w_i_i + 1j * x_w_i_r  # x_w_i_C: b x M
            W.append(x_w_i_C)
        W = torch.stack(W, dim=-1)  # W: b x M x K

        # Power normalization layer
        W_norm = torch.sum(torch.linalg.vector_norm(W, dim=1).pow(2), 1)  # W_norm: b x 1
        W_norm_fac = torch.sqrt(1 / W_norm)
        W = W * W_norm_fac.unsqueeze(-1).unsqueeze(-1)  # W: b x M x K

        # Compute sum rate
        rate_list = []
        for i in range(self.n_ue):
            h_i = x[:, i, :].unsqueeze(1)  # h_i: b x 1 x M
            W_i = W[:, :, i].unsqueeze(2)  # W_i: b x M x 1
            W_i_int = torch.cat((W[:, :, :i], W[:, :, i + 1:]), dim=2)  # W_i_int: b x M x (K-1)

            # Get numerator
            P_i = (torch.matmul(h_i, W_i).squeeze()).abs().pow(2)  # Numerator: 1 x b

            # Get denominator
            Int_i = torch.sum(torch.matmul(h_i, W_i_int).squeeze(1).abs().pow(2), dim=-1)  # Int_i: b x 1

            SINR = P_i / (n_power + Int_i)

            # Get rate
            rate_i = torch.log2(1 + SINR).squeeze().mean()
            rate_list.append(rate_i)

        sum_rate = sum(rate_list)
        rate_list.append(sum_rate)

        return W, vq_loss, rate_list
