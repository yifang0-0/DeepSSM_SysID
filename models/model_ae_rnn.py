import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as tdist
from torchsummary import summary


import roboticstoolbox as rtb
"""implementation of the Variational Auto Encoder Recurrent Neural Network (VAE-RNN) from 
https://backend.orbit.dtu.dk/ws/portalfiles/portal/160548008/phd475_Fraccaro_M.pdf and partly from
https://arxiv.org/pdf/1710.05741.pdf using unimodal isotropic gaussian distributions for inference, prior, and 
generating models."""


class AE_RNN(nn.Module):
    def __init__(self, param, device,  sys_param={},  dataset="toy_lgssm", bias=False):
        super(AE_RNN, self).__init__()

        self.y_dim = param.y_dim
        self.u_dim = param.u_dim
        self.h_dim = param.h_dim
        self.z_dim = param.z_dim
        self.n_layers = param.n_layers
        self.device = device
        self.mpnt_wt = param.mpnt_wt
        self.param = sys_param
        self.dataset = dataset
        # print("self.device", self.device)
        # print(self.param['A_prt'], self.param['B_prt'],self.param['C'],self.mpnt_wt)


        # feature-extracting transformations (phi_y, phi_u and phi_z)
        self.phi_u = nn.Sequential(
            nn.Linear(self.u_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),)
        self.phi_x = nn.Sequential(
            nn.Linear(self.z_dim+self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),)
        
        self.x_mean = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),)
        
        self.x_logvar = nn.Sequential(
            nn.Linear(self.h_dim, self.z_dim),
            # nn.ReLU()
            )

        # encoder function (phi_enc) -> Inference
        self.dynn = nn.Sequential(
            nn.Linear(self.h_dim + self.h_dim, self.h_dim),
            # nn.Dropout(),
            # nn.Tanh(),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            # nn.ReLU(),
           )
        
        self.menn = nn.Sequential(
            nn.Linear(self.h_dim, self.h_dim),
            # nn.Dropout(),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.y_dim),
            # nn.ReLU(),
            )

        # recurrence function (f_theta) -> Recurrence
        self.rnn = nn.GRU(self.h_dim, self.h_dim, self.n_layers, bias)
        
    def dynamic_phy(self, u_tm1, z_tm1):
        batch_size = u_tm1.shape[0]
        if "lgssm" in self.dataset:
            A_prt =torch.tensor(self.param['A_prt'], dtype=torch.float32,device=self.device)
            A_prt = A_prt.expand(batch_size, -1, -1)

            B_prt =torch.tensor(self.param['B_prt'], dtype=torch.float32,device=self.device)
            B_prt = B_prt.expand(batch_size, -1, -1)
            z_phy_t =  torch.matmul(A_prt,z_tm1.unsqueeze(-1)).squeeze(-1)+torch.matmul(B_prt,u_tm1.unsqueeze(-1)).squeeze(-1)
        elif self.dataset == "industrobo":
            # 
            z_phy_t = 0
        return z_phy_t
    
    def dynamic_phy_z(self, z_tm1,u_tm1=0):
        if "lgssm" in self.dataset:
            batch_size = z_tm1.shape[0]
            C =torch.tensor(self.param['C'], dtype=torch.float32,device=self.device)
            C = C.expand(batch_size, -1, -1)

            z_phy_t =  torch.matmul(C,z_tm1.unsqueeze(-1)).squeeze(-1)
        return z_phy_t
    
    def measure_phy(self, z_mean_t, z_logvar_t):
        if "lgssm" in self.dataset:
            C = torch.tensor( self.param['C'], dtype=torch.float32,device=self.device)
            sigma = torch.tensor( self.param['sigma_out'], dtype=torch.float32,device=self.device)
            sigma2 = torch.pow(sigma,2)
            z_logvar_t = z_logvar_t.unsqueeze(-1)
            z_var_t = torch.matmul(C*(z_logvar_t.exp()),C.T)
            measure_mean = torch.matmul(C,z_mean_t.unsqueeze(-1)).squeeze(-1)
            measure_var = (z_var_t+sigma2).squeeze(-1)                     
        return measure_mean, measure_var
    

    def forward(self, u, y):
        #  batch size
        batch_size = y.shape[0]
        seq_len = y.shape[2]

        # allocation
        loss = 0
        # initialization
        h = torch.rand(self.n_layers, batch_size, self.h_dim, device=self.device)

        # for all time steps
        for t in range(seq_len):
            # print("phi_u.is_cuda()", u[:, :, t].get_device())

            # feature extraction: u_t
            phi_u_t = self.phi_u(u[:, :, t])

            dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
            x_mean = self.x_mean(dynn_phi)
            x_logvar = self.x_logvar(dynn_phi)
            
            # recurrence: u_t+1 -> h_t+1
            _, h = self.rnn(phi_u_t.unsqueeze(0), h)

            
            
            # print("all_loss", loss)
            if self.mpnt_wt>100:
                param_C =  torch.tensor(self.param['C'], dtype=torch.float32,device=self.device)
                y_hat_mean = torch.matmul(param_C,x_mean.t())
                var_sigma= torch.pow(torch.tensor(self.param['sigma_out'], dtype=torch.float32,device=self.device),2)
                var_x = torch.matmul(param_C*x_logvar.exp(),param_C.T)
                y_var_y = var_sigma+var_x
                y_pred = tdist.Normal(y_hat_mean.T, (y_var_y).sqrt())
                y_hat = y_pred.rsample()
                loss += torch.sum((y_hat-y[:, :, t]) ** 2)
                # if t==20:
                #     print("y_hat_mean.shape, y_hat.shape, y[:,:,t].shape ",y_hat_mean.shape, y_hat.shape, y[:,:,t].shape)
                #     print("y_pred: ",y_pred)
                #     print("(y_var_y).sqrt().shape ",(y_var_y).sqrt().shape)
                    



            elif self.mpnt_wt>=10:
                phi_x = self.phi_x(torch.cat([x_mean, x_logvar], 1))
                y_hat_nn = self.menn(phi_x)
                y_hat_phy = self.dynamic_phy_z(x_mean)
                y_hat = y_hat_nn+y_hat_phy
                loss += torch.sum((y_hat-y[:, :, t]) ** 2)
                
            elif self.mpnt_wt<=0:
                #pure nn
                phi_x = self.phi_x(torch.cat([x_mean, x_logvar], 1))
                y_hat = self.menn(phi_x)
                loss += torch.sum((y_hat-y[:, :, t]) ** 2)
            else:               
                ## add measurement known panalty
                phi_x = self.phi_x(torch.cat([x_mean, x_logvar], 1))
                y_hat = self.menn(phi_x)
                measure_mean_t, measure_var_t = self.measure_phy(x_mean, x_logvar )
                pred_measurepanalty_dist = tdist.Normal(measure_mean_t, measure_var_t.sqrt())
                loss_panelty = torch.sum(pred_measurepanalty_dist.log_prob(y[:, :, t]))
                loss += (torch.sum((y_hat-y[:, :, t]) ** 2) - self.mpnt_wt*loss_panelty)
                
                
                
        return loss

    def generate(self, u):
        # get the batch size
        batch_size = u.shape[0]
        # length of the sequence to generate
        seq_len = u.shape[-1]
        y_hat = torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)
        y_hat_sigma =  torch.zeros(batch_size, self.y_dim, seq_len, device=self.device)

        x = torch.zeros(batch_size, self.z_dim, seq_len, device=self.device)
        h = torch.rand(self.n_layers, batch_size, self.h_dim, device=self.device)

        print("mpnt_wt: ",self.mpnt_wt)
        # for all time steps
        for t in range(seq_len):

            # feature extraction: u_t
            phi_u_t = self.phi_u(u[:, :, t])

            dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
            x[:,:,t] = self.x_mean(dynn_phi)
            x_logvar = self.x_logvar(dynn_phi)

            # phi_x = self.phi_x(x[:,:,t])
            # y_hat[:, :, t] = self.menn(phi_x)
            
            
            # recurrence: u_t+1 -> h_t+1
            _, h = self.rnn(phi_u_t.unsqueeze(0), h)
                        # print("all_loss", loss)
            if self.mpnt_wt>100:

                # hard constraints
                # y_var = cx_varc_sigma
                param_C =  torch.tensor(self.param['C'], dtype=torch.float32,device=self.device)
                y_hat_mean = torch.matmul(param_C, x[:,:,t].t())
                y_var_y = torch.pow(torch.tensor(self.param['sigma_out'], dtype=torch.float32,device=self.device),2) + torch.matmul(param_C*x_logvar.exp(),param_C.T)
                y_pred = tdist.Normal(y_hat_mean.T, (y_var_y).sqrt())
                y_hat[:, :, t] = tdist.Normal.rsample(y_pred)
                
                        
            elif self.mpnt_wt>=10:
                
                # physical augmentation CX
                phi_x = self.phi_x(torch.cat([x[:,:,t], x_logvar], 1))
                y_hat_nn = self.menn(phi_x)
                y_hat_phy = self.dynamic_phy_z(x[:,:,t])
                y_hat[:, :, t] = y_hat_nn+y_hat_phy
                # y_hat_sigma[:, :, t] =  measure_var_t.sqrt()

            elif self.mpnt_wt<=0:
                # pure nn
                # remember to add y_var (maybe not)
                phi_x = self.phi_x(torch.cat([x[:,:,t], x_logvar], 1))
                y_hat[:, :, t] = self.menn(phi_x)
                
            else:
                
                # panelty
                phi_x = self.phi_x(torch.cat([x[:,:,t], x_logvar], 1))
                y_hat[:, :, t]  = self.menn(phi_x)

         
        y_hat_mu = y_hat

        return y_hat, y_hat_mu, y_hat_sigma, x

