import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as tdist
import numpy as np

'''
Encoder:

    NN:
    ut →ht →xt<-yt
            ↑
            ht-1

    phy-NN:
    ut-1    →               
    xt-1    →  xt-phy
                     + xt ← yt
    ut → ht →  xt-nn
    
    
Decoder:

    NN:
    ut →ht →xt → yt
            ↑
            ht-1

    phy-NN:
    ut-1    →              
    xt-1    → xt-phy
                     + xt → yt
    ut → ht → xt-nn

Where to add physical information?
Maybe module: phy-x
'''

class VAE_RNN_PHYNN(nn.Module):
    def __init__(self, param, device, sys_param={}, enc_index=0, dec_index=0, bias=False):
        super(VAE_RNN_PHYNN, self).__init__()

        self.y_dim = param.y_dim
        self.u_dim = param.u_dim
        self.h_dim = param.h_dim
        self.z_dim = param.z_dim
        self.mpnt_wt = param.mpnt_wt
        self.param = sys_param
        
        self.n_layers = param.n_layers
        self.device = device

        print(self.param['A_prt'], self.param['B_prt'],self.param['C'],self.mpnt_wt)
        
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
            # nn.ReLU(),
            )

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
            # nn.ReLU()
            )

        # decoder function (phi_dec) -> Generation
        self.dec = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU())
        self.dec_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.y_dim),)
        self.dec_logvar = nn.Sequential(
            nn.Linear(self.h_dim, self.y_dim),
            # nn.ReLU()
            )

        
        # recurrence function (f_theta) -> Recurrence
        self.rnn = nn.GRU(self.h_dim, self.h_dim, self.n_layers, bias)
    
    # New z_phy_t = A'zt-1+B'ut-1
    def dynamic_phy_linear_z(self, u_tm1, z_tm1):
        batch_size = u_tm1.shape[0]
        A_prt =torch.tensor(self.param['A_prt'], dtype=torch.float32,device=self.device)
        A_prt = A_prt.expand(batch_size, -1, -1)

        B_prt =torch.tensor(self.param['B_prt'], dtype=torch.float32,device=self.device)
        B_prt = B_prt.expand(batch_size, -1, -1)
        # print("A_prt.shape",A_prt.shape)
        # print("B_prt.shape",B_prt.shape)
        # print("z_tm1.shape",z_tm1.shape)
        # print("u_tm1.shape",u_tm1.shape)
        # print("A_prt,z_tm1.unsqueeze(-1)", torch.matmul(A_prt,z_tm1.unsqueeze(-1)).shape)
        # print("B_prt,u_tm1.unsqueeze(-1)", torch.matmul(B_prt,u_tm1.unsqueeze(-1)).shape)
        z_phy_t =  torch.matmul(A_prt,z_tm1.unsqueeze(-1)).squeeze(-1)+torch.matmul(B_prt,u_tm1.unsqueeze(-1)).squeeze(-1)
        return z_phy_t

    # New prior(mean, var) is according from y=C(x+x_phy)+e, 

    # def prior_phy_linear_y(self, z_t):
    #     C = torch.tensor( self.param['C'], dtype=torch.float32,device=self.device)
    #     prior_mean = torch.matmul(torch.inverse(C),y_t)
    #     y_var = torch.tensor(self.param['w'], dtype=torch.float32,device=self.device)
    #     prior_var = torch.matmul(torch.inverse(C),torch.matmul(y_var,torch.inverse(C)))
    #     prior_log_var = torch.log(prior_var)
                                
    #     return prior_mean, prior_log_var


    def measure_phy_linear_y(self, z_mean_t, z_logvar_t):
        batch_size = z_mean_t.shape[0]
        # input matrix and measurement noise from the setting file
        C = torch.tensor( self.param['C'], dtype=torch.float32,device=self.device)

        sigma = torch.tensor( self.param['sigma_out'], dtype=torch.float32,device=self.device)
        sigma2 = torch.pow(sigma,2)
        # C.shape torch.Size([31, 1, 2]) after expansion
        # C = C.expand(batch_size, -1, -1)
        # z_logvar_t.shape torch.Size([31, 2]) -> [31,2,1]
        z_logvar_t = z_logvar_t.unsqueeze(-1)
        
        # new var=sigma.+C-t*logvar.exp()*C
        # new mean = C*mean
        # print("z_logvar_t.exp() is !!!!!!!",z_logvar_t.exp())
        # print("C*z_logvar_t.exp() is !!!!!!!",z_logvar_t.shape,torch.matmul(C,(C.T*(z_logvar_t.exp()))).shape,C.shape)

        
        # print("C.T*z_logvar_t.exp()*C is !!!!!!!",torch.matmul((C.T*z_logvar_t.exp()),C))
        
        

        z_var_t = torch.matmul(C,C.T*(z_logvar_t.exp()))

        measure_mean = (torch.matmul(C.unsqueeze(0).expand(batch_size,-1,-1),z_mean_t.unsqueeze(-1))).squeeze(-1)
        measure_var = (z_var_t+sigma2).squeeze(-1)                     
        # print("measure_mean,measure_var",measure_mean.shape,measure_var.shape)
        # print("z_mean_t,z_logvar_t",z_mean_t.shape,z_logvar_t.shape)
        
        # print("prior_log_var.shape",prior_log_var.shape)
        
        # print("z_var_t,sigma2")
        
        # print(z_var_t,C.shape, z_logvar_t.exp(),z_logvar_t.exp().squeeze(-1).shape)
        
        return measure_mean, measure_var
        
    
    def forward(self, u, y):
        #  batch size
        batch_size = y.shape[0]
        seq_len = y.shape[2]

        # allocation
        loss = 0
        # initialization
        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        t_index = 0
        # for all time steps
        for t in range(seq_len):
            if t_index == 0:
                t_index = 1
                z_tm1 = torch.zeros(batch_size, self.z_dim, device=self.device)
                z_phy_t = torch.zeros(batch_size, self.z_dim, device=self.device)
                # physical part of nn is
            else:
                z_tm1 = z_t
                z_phy_t = self.dynamic_phy_linear_z(u[:, :, t-1], z_tm1)
            
            # feature extraction: y_t
            phi_y_t = self.phi_y(y[:, :, t])
            # feature extraction: u_t
            phi_u_t = self.phi_u(u[:, :, t])

            
            # it is different for encoder_mean z and decoder mean z: in forward training, the kld based on physics added z_mean
            # (so we use self.enc_mean(enc_t)+z_phy_t)
            # but in the generation, prior_mean is the mean distribution of z, z_t was used to generate y_mean and y_var, 
            #  so here we use z_t for more robustness, (same in dec_t = param_C*z_t )
            # encoder: y_t, h_t -> z_t
            # z = phy+nn
            enc_t = self.enc(torch.cat([phi_y_t, h[-1]], 1))
            enc_mean_t = self.enc_mean(enc_t)+z_phy_t
            enc_logvar_t = self.enc_logvar(enc_t)

            # sampling and reparameterization: get a new z_t
            temp = tdist.Normal(enc_mean_t, enc_logvar_t.exp().sqrt())
            z_nn_t = tdist.Normal.rsample(temp)

            z_t = z_nn_t
            
            # feature extraction: z_t
            phi_z_t = self.phi_z(z_t)

            # prior: h_t -> z_t
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)
            
            # let's say we use hard constraints here where we see no decoder needs to be trained
            if self.mpnt_wt>100:
                param_C =  torch.tensor(self.param['C'], dtype=torch.float32,device=self.device)
                
                # dec_t = torch.matmul(param_C,z_t.t()) # this is the one with larger loss
                dec_t = torch.matmul(z_t,param_C.t()) # smaller oss (from vae-rnn-phy)
                # print("z_t,dec_t",z_t,dec_t)
                # dec_t = param_C*z_t 
                dec_mean_t = dec_t
                dec_var_t = torch.pow(torch.tensor(self.param['sigma_out'], dtype=torch.float32,device=self.device),2) 
                z_var_t = torch.matmul(param_C*enc_logvar_t.exp(),param_C.T) 
                pred_dist = tdist.Normal(dec_mean_t, (dec_var_t+z_var_t).sqrt())


                # computing the loss
                KLD = self.kld_gauss(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
                loss_pred = torch.sum(pred_dist.log_prob(y[:, :, t]))
                loss += - loss_pred + KLD
                # print(dec_mean_t,(dec_var_t+z_var_t).sqrt())
                # print("KLD,loss_pred,z_var_t",KLD,loss_pred,z_var_t)
                # print(pred_dist)
            elif self.mpnt_wt>=10:
     
                # decoder: z_t -> y_t
                dec_t = self.dec(phi_z_t)
                measure_mean_t, measure_var_t = self.measure_phy_linear_y( z_t, enc_logvar_t )
                dec_mean_t = self.dec_mean(dec_t)+measure_mean_t
                dec_logvar_t = torch.log(measure_var_t+self.dec_logvar(dec_t).exp())
                pred_dist = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt())

                # computing the loss
                KLD = self.kld_gauss(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
                loss_pred = torch.sum(pred_dist.log_prob(y[:, :, t]))
                loss += - loss_pred + KLD
                
            else:
                # decoder: z_t -> y_t
                dec_t = self.dec(phi_z_t)
                dec_mean_t = self.dec_mean(dec_t)
                dec_logvar_t = self.dec_logvar(dec_t)
                pred_dist = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt())

                # computing the loss
                KLD = self.kld_gauss(enc_mean_t, enc_logvar_t, prior_mean_t, prior_logvar_t)
                loss_pred = torch.sum(pred_dist.log_prob(y[:, :, t]))
                # loss += - loss_pred + KLD 
                
                ## add measurement known panalty
                measure_mean_t, measure_var_t = self.measure_phy_linear_y( z_t, enc_logvar_t )
                pred_measurepanalty_dist = tdist.Normal(measure_mean_t, measure_var_t.sqrt())
                # loss_panelty = torch.sum(pred_measurepanalty_dist.log_prob(y[:, :, t]))           
                # loss += (- loss_pred + KLD - self.mpnt_wt*loss_panelty)
                loss_panelty = torch.sum(pred_measurepanalty_dist.log_prob(y[:, :, t]))
                loss += (- loss_pred + KLD -  self.mpnt_wt*loss_panelty)
                # recurrence: u_t+1 -> h_t+1
                
            _, h = self.rnn(phi_u_t.unsqueeze(0), h)

        return loss

    def generate(self, u):
        # get the batch size
        batch_size = u.shape[0]
        # length of the sequence to generate
        seq_len = u.shape[-1]

        print("batch size:" ,batch_size)
        # allocation
        sample = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        sample_mu = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        sample_sigma = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        z = torch.zeros(batch_size, self.z_dim, seq_len, device=self.device)

        h = torch.zeros(self.n_layers, batch_size, self.h_dim, device=self.device)
        t_index = 0
        
        # for all time steps
        for t in range(seq_len):
            if t_index == 0:
                z_tm1 = torch.zeros(batch_size, self.z_dim, device=self.device)
                z_phy_t = torch.zeros(batch_size, self.z_dim, device=self.device)
                # physical part of nn is
            else:
                z_tm1 = z_t
                t_index = 1
                z_phy_t = self.dynamic_phy_linear_z(u[:, :, t-1], z_tm1)
            # feature extraction: u_t+1
            phi_u_t = self.phi_u(u[:, :, t])

            # prior: h_t -> z_t
            prior_t = self.prior(h[-1])
            prior_mean_t = self.prior_mean(prior_t)
            prior_logvar_t = self.prior_logvar(prior_t)

            # sampling and reparameterization: get new z_nn_t
            temp_z_nn = tdist.Normal(prior_mean_t, prior_logvar_t.exp().sqrt())
            z_nn_t = tdist.Normal.rsample(temp_z_nn)
            z_t = z_nn_t+z_phy_t
            z[:,:,t] = z_t
            # feature extraction: z_nn_t

                        # let's say we use hard constraints here where we see no decoder needs to be trained
            if self.mpnt_wt>100:
                param_C =  torch.tensor(self.param['C'], dtype=torch.float32,device=self.device)
                dec_t = torch.matmul(z_t,param_C.t())

                dec_mean_t = dec_t
                dec_var_t = torch.pow(torch.tensor(self.param['sigma_out'], dtype=torch.float32,device=self.device),2) 

                z_var_t = torch.matmul(param_C*prior_logvar_t.exp(),param_C.T)
                temp = tdist.Normal(dec_mean_t, (dec_var_t+z_var_t).sqrt())
                sample[:, :, t] = tdist.Normal.rsample(temp)
                
                # store mean and std
                sample_mu[:, :, t] = dec_mean_t
                sample_sigma[:, :, t] = (dec_var_t+z_var_t).sqrt()
                                # print("z_var_t,dec_var_t,z_var_t,shape,dec_var_t.shape",z_var_t,dec_var_t,z_var_t.shape,dec_var_t.shape,)
                # temp = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt()+z_var_t)
                                # torch.matmul(C,C.T*(z_logvar_t.exp()))
                                                # print("dec_t.shape,param_C.shape,z_t.shape \n",dec_t.shape,param_C.shape,z_t.shape)
            elif self.mpnt_wt>=10:
                phi_z_t = self.phi_z(z_t)
                # decoder: z_t -> y_t
                dec_t = self.dec(phi_z_t)
                measure_mean_t, measure_var_t = self.measure_phy_linear_y( z_t, prior_logvar_t )
                dec_mean_t = self.dec_mean(dec_t)+measure_mean_t
                dec_logvar_t = torch.log(measure_var_t+self.dec_logvar(dec_t).exp())
                # print("measure_var_t+self.dec_logvar(dec_t).exp()",measure_var_t.shape,(self.dec_logvar(dec_t).exp()).shape)
                # print("measure_var_t+self.self.dec_mean(dec_t).exp()",measure_mean_t.shape,(self.dec_mean(dec_t)).shape)
                
                # store the samples
                temp = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt())
                # print(measure_var_t,self.dec_logvar(dec_t).exp())
                # print(dec_mean_t,dec_logvar_t.exp().sqrt(),tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt()),sample)
                sample[:, :, t] = tdist.Normal.rsample(temp)
                # store mean and std
                sample_mu[:, :, t] = dec_mean_t
                sample_sigma[:, :, t] = dec_logvar_t.exp().sqrt()


               
            else: 
                phi_z_t = self.phi_z(z_t)

                # decoder: z_t -> y_t
                dec_t = self.dec(phi_z_t)
                dec_mean_t = self.dec_mean(dec_t)
                dec_logvar_t = self.dec_logvar(dec_t)
                # store the samples
                temp = tdist.Normal(dec_mean_t, dec_logvar_t.exp().sqrt())
                sample[:, :, t] = tdist.Normal.rsample(temp)
                # store mean and std
                sample_mu[:, :, t] = dec_mean_t
                sample_sigma[:, :, t] = dec_logvar_t.exp().sqrt()
                # recurrence: u_t+1 -> h_t+1
            
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
