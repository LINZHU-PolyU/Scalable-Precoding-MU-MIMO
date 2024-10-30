import torch
import torch.nn as nn


class Pilot_Network(nn.Module):
    def __init__(self, M, UE_antenna, L):
        super(Pilot_Network, self).__init__()
        self.M = M  # M
        self.UE_antenna = UE_antenna  # K
        self.L = L  # L

        W_real = torch.empty(self.M, self.L)  # M x L
        W_imag = torch.empty(self.M, self.L)  # M x L

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
    def __init__(self, x_dim, D):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x_dim, 1024),  # x_dim: dimension of input = 2L
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, D),  # D: dimension of extracted feature
            # nn.Tanh()
        )

    def forward(self, x):
        return self.encoder(x)


class BF_Network(nn.Module):
    def __init__(self, D, K, M):
        super(BF_Network, self).__init__()
        self.K = K  # K
        self.M = M  # M

        self.decoder = nn.Sequential(
            nn.Linear(D * K, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
        )

        dim_out = K * M
        self.bf_real = nn.Linear(512, dim_out)
        self.bf_imag = nn.Linear(512, dim_out)

    def forward(self, x):
        x = self.decoder(x)
        w_real = self.bf_real(x)  # w_real: b x KM
        w_imag = self.bf_imag(x)  # w_imag: b x KM

        # Reshape w_real, w_imag
        w_real = w_real.view(-1, self.M, self.K)  # w_real: b x M x K
        w_imag = w_imag.view(-1, self.M, self.K)  # w_imag: b x M x K
        W = w_real + 1j * w_imag  # w: b x M x K

        return W


class baselineModel(nn.Module):
    def __init__(self, K, M, UE_ant, B, L, SNR):
        super(baselineModel, self).__init__()
        self.pilot = Pilot_Network(M, UE_ant, L)
        self.SNR = SNR
        x_dim = 2 * L  # 2L

        self.encoder = nn.ModuleList([encoder(x_dim, B) for _ in range(K)])  # Each UE has its own encoder
        self.sigmoid = nn.Sigmoid()
        self.BF_Network = BF_Network(B, K, M)  # Beamforming Network

    def forward(self, x, alpha):
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
        for i in range(len(self.encoder)):
            pilot_real_i = pilot_real_noisy[:, i, :].squeeze()  # pilot_real_i: b x L
            pilot_imag_i = pilot_imag_noisy[:, i, :].squeeze()  # pilot_imag_i: b x L
            pilot_i = torch.cat((pilot_real_i, pilot_imag_i), dim=-1)  # pilot_i: b x 2L

            # Encoder
            encoder_i_res = self.encoder[i](pilot_i)  # encoder_res: b x B
            encoder_i_res_tanh = 2 * self.sigmoid(alpha * encoder_i_res) - 1
            enc_i_sign = torch.sign(encoder_i_res)  # enc_i_sign: b x B
            enc_i_res = encoder_i_res_tanh + (enc_i_sign - encoder_i_res_tanh).detach()  # enc_i_res: b x B

            encoder_res.append(enc_i_res)

        # Concatenate encoder results
        encoder_res_cat = torch.cat(encoder_res, dim=-1)  # encoder_res_cat: b x (K * B)

        # BF Network
        W = self.BF_Network(encoder_res_cat)  # W: b x M x K

        # Power normalization layer
        W_norm = torch.sum(torch.linalg.vector_norm(W, dim=1).pow(2), 1)  # W_norm: b x 1
        W_norm_fac = torch.sqrt(1 / W_norm)
        W = W * W_norm_fac.unsqueeze(-1).unsqueeze(-1)  # W: b x M x K

        # Compute sum rate
        rate_list = []
        for i in range(len(self.encoder)):
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

        return W, rate_list
