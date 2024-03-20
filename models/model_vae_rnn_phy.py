import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as tdist
import numpy as np
"""implementation of the Variational Auto Encoder Recurrent Neural Network (VAE-RNN) from 
https://backend.orbit.dtu.dk/ws/portalfiles/portal/160548008/phd475_Fraccaro_M.pdf and partly from
https://arxiv.org/pdf/1710.05741.pdf using unimodal isotropic gaussian distributions for inference, prior, and 
generating models."""


class VAE_RNN_PHY(nn.Module):
    def __init__(self, param, device, sys_param={}, bias=False):
        super(VAE_RNN_PHY, self).__init__()

        self.y_dim = param.y_dim
        self.u_dim = param.u_dim
        self.h_dim = param.h_dim
        self.z_dim = param.z_dim
        self.n_layers = param.n_layers
        self.device = device
        self.sys_param = sys_param
        print("self.sys_param['C']",self.sys_param['C'],"self.sys_param['np_log_out']",np.log(self.sys_param['sigma_out']))
        # feature-extracting transformations (phi_y, phi_u and phi_z)
        self.phi_y = nn.Sequential(
            nn.Linear(self.y_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),)
        self.phi_u = nn.Sequential(
            nn.Linear(self.u_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),)
        self.phi_z = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),)

        # encoder function (phi_enc) -> Inference
        self.enc = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),)
        self.enc_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim))
        self.enc_logvar = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.ReLU(),)

        # prior function (phi_prior) -> Prior
        self.prior = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),)
        self.prior_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim))
        self.prior_logvar = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            nn.ReLU())

        # recurrence function (f_theta) -> Recurrence
        self.rnn = nn.GRU(self.h_dim, self.h_dim, self.n_layers, bias)

    def forward(self, u, y):
        #  batch size
        batch_size = y.shape[0]
        seq_len = y.shape[2]

        # allocation
        loss = 0
        # initialization
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)

        # for all time steps
        for t in range(seq_len):
            # feature extraction: y_t
            phi_y_t = self.phi_y(y[:, :, t])
            # feature extraction: u_t
            phi_u_t = self.phi_u(u[:, :, t])

            # encoder: y_t, h_t -> z_t
            enc_t = self.enc(torch.cat([phi_y_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)
            enc_logvar_t = self.enc_logvar(enc_t)

            # prior: h_t -> z_t (for KLD loss)
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)

            # sampling and reparameterization: get a new z_t
            temp = tdist.Normal(enc_mean_t, enc_logvar_t.exp().sqrt())
            z_t = tdist.Normal.rsample(temp)

            # decoder: z_t -> y_t
            # decoder function = C*Z+e
            # constant_matrix = torch.tensor([[1, 0]], dtype=torch.float32, device=self.device).t()
            # self.constant_matrix = nn.Parameter(constant_matrix, requires_grad=False)
        
            dec_t = torch.matmul(z_t, torch.tensor(self.sys_param['C'], dtype=torch.float32,device=self.device).t())
            
            dec_mean_t = dec_t
            dec_logvar_t = torch.tensor(np.log(self.sys_param['sigma_out']), dtype=torch.float32,device=self.device)
            z_var_t = torch.matmul(enc_logvar_t.exp().sqrt(), torch.tensor(self.sys_param['C'], dtype=torch.float32,device=self.device).t())
            pred_dist = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt()+z_var_t)

            # recurrence: u_t+1 -> h_t+1
            _, h = self.rnn(phi_u_t.unsqueeze(0), h)

            # computing the loss
            KLD = self.kld_gauss(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
            loss_pred = torch.sum(pred_dist.log_prob(y[:, :, t]))
            loss += - loss_pred + KLD

        return loss

    def generate(self, u):
        # get the batch size
        batch_size = u.shape[0]
        # length of the sequence to generate
        seq_len = u.shape[-1]

        # allocation
        sample = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        sample_mu = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        sample_sigma = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        
        z = torch.zeros(batch_size, self.z_dim, seq_len, device=self.device)
        

        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)

        # for all time steps
        for t in range(seq_len):
            # feature extraction: u_t+1
            phi_u_t = self.phi_u(u[:, :, t])

            # prior: h_t -> z_t
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)

            # sampling and reparameterization: get new z_t
            temp = tdist.Normal(prior_mean_t, prior_logvar_t.exp().sqrt())
            z_t = tdist.Normal.rsample(temp)
            
            z[:,:,t] =z_t
            # decoder: z_t -> y_t
            # decoder function = C*Z+e
            # constant_matrix = torch.tensor([[1, 0]], dtype=torch.float32,device=self.device).t()
            # self.constant_matrix = nn.Parameter(constant_matrix, requires_grad=False)
        
            dec_t = torch.matmul(z_t, torch.tensor(self.sys_param['C'], dtype=torch.float32,device=self.device).t())
            dec_mean_t = dec_t
            dec_logvar_t = torch.tensor(self.sys_param['sigma_out'], dtype=torch.float32,device=self.device)
            z_var_t = torch.matmul(prior_logvar_t.exp().sqrt(), torch.tensor(self.sys_param['C'], dtype=torch.float32,device=self.device).t())

            # store the samples
            temp = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt()+z_var_t)
            sample[:, :, t] = tdist.Normal.rsample(temp)
            
            # store mean and std
            sample_mu[:, :, t] = dec_mean_t
            sample_sigma[:, :, t] = dec_logvar_t.exp().sqrt()+z_var_t

            # recurrence: u_t+1, z_t -> h_t+1
            _, h = self.rnn(phi_u_t.unsqueeze(0), h)

        return sample, sample_mu, sample_sigma, z

    @staticmethod
    def kld_gauss(mu_q, logvar_q, mu_p, logvar_p):
        # Goal: Minimize KL divergence between q_pi(z|xi) || p(z|xi)
        # This is equivalent to maximizing the ELBO: -D_KL(q_phi(z|xi) || p(z)) + Reconstruction term
        # This is equivalent to minimizing D_KL(q_phi(z|xi) || p(z))
        term1 = logvar_p - logvar_q - 1
        term2 = (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
        kld = 0.5 * torch.sum(term1 + term2)

        return kld