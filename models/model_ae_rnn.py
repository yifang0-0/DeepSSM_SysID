import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.distributions as tdist
from torchsummary import summary
from models.physical_augment.model_phy import MODEL_PHY

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
        self.phy_aug = MODEL_PHY(self.dataset, self.param, self.device)
        # print("self.device", self.device)
        # print(self.param['A_prt'], self.param['B_prt'],self.param['C'],self.mpnt_wt)


        # feature-extracting transformations (phi_y, phi_u and phi_z)
        self.phi_u = nn.Sequential(
            nn.Linear(self.u_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),)
        self.phi_x = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
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
        if self.mpnt_wt <=-10: 
            self.dynn_phy = nn.Sequential(
                nn.Linear(self.h_dim + self.h_dim +self.h_dim, self.h_dim),
                # nn.Dropout(),
                # nn.Tanh(),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),
                # nn.ReLU(),
               )
            self.menn_phy = nn.Sequential(
                nn.Linear(self.h_dim+self.h_dim, self.h_dim),
                # nn.Dropout(),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.y_dim),
                # nn.ReLU(),
                )
            self.x_phi_phy = nn.Sequential(
                nn.Linear(self.z_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),)
            
            self.y_phi_phy = nn.Sequential(
                nn.Linear(self.y_dim, self.h_dim),
                nn.ReLU(),
                nn.Linear(self.h_dim, self.h_dim),)
            

        # recurrence function (f_theta) -> Recurrence
        self.rnn = nn.GRU(self.h_dim, self.h_dim, self.n_layers, bias)
        
    def dyphy(self, u, x):

        x_phy_t = self.phy_aug.dynamic_model(u,x)  

        return x_phy_t
    
    def mephy(self, u, x):
        y_phy_t = self.phy_aug.measurement_model(u,x)               
        return y_phy_t
    
    

    

    def forward(self, u, y):
        #  batch size
        torch.autograd.set_detect_anomaly(True)
        batch_size = y.shape[0]
        seq_len = y.shape[2]

        # allocation
        loss = 0
        # initialization
        h = torch.rand(self.n_layers, batch_size, self.h_dim, dtype=torch.float32,device=self.device)
        
        x = torch.zeros(batch_size, self.z_dim, seq_len, dtype=torch.float32, device=self.device)
        

        # for all time steps
        for t in range(seq_len):
            # print("seq no.: ", t)
            # print("phi_u.is_cuda()", u[:, :, t].get_device())
            if t == 0:
                x_tm1 =  x[:,:,t].clone()
            else: 
                x_tm1 = x[:,:,t-1].clone()

            # feature extraction: u_t
            phi_u_t = self.phi_u(u[:, :, t])
            if self.mpnt_wt>100:
                # pure physical
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1)
                x_t = x_mean_phy

            elif  self.mpnt_wt>=10:
                #physics augmented
                dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
                x_mean_nn = self.x_mean(dynn_phi)
                x_logvar = self.x_logvar(dynn_phi)
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1)
                x_t =  x_mean_nn + x_mean_phy
                
                
            elif self.mpnt_wt<=-10:
                #physics guided
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1)
                if x_mean_phy.dtype != self.x_phi_phy[0].weight.dtype:
                    x_mean_phy_f32 = x_mean_phy.to(self.x_phi_phy[0].weight.dtype)
                    x_phy_phi = self.x_phi_phy(x_mean_phy_f32)
                else:
                    x_phy_phi = self.x_phi_phy(x_mean_phy)
                dynn_phi = self.dynn_phy(torch.cat([phi_u_t, h[-1],x_phy_phi], 1))
                x_mean_nn = self.x_mean(dynn_phi)
                x_t = x_mean_nn
            elif self.mpnt_wt<=0:
                dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
                x_mean_nn = self.x_mean(dynn_phi)
                x_t = x_mean_nn
            
            #save x_t
            x[:,:,t] = x_t
            # recurrence: u_t+1 -> h_t+1
            _, h = self.rnn(phi_u_t.unsqueeze(0), h)

            
            
            # print("all_loss", loss)

            if self.mpnt_wt>100:
                # pure physical constraints
                y_hat_phy = self.mephy(u[:,:,t],x_t)
                y_hat[:, :, t] = y_hat_phy



            elif self.mpnt_wt>=10:
                #physics augmented
                # phi_x_t = self.phi_x(torch.cat([x_t, x_logvar], 1))
                phi_x_t = self.phi_x(x_t)
                y_hat_nn = self.menn(phi_x_t)
                y_hat_phy = self.mephy(u[:,:,t], x_t)
                y_hat = y_hat_nn+y_hat_phy
                loss += torch.sum((y_hat-y[:, :, t]) ** 2)
                
            elif self.mpnt_wt<=-10:
                #physics guided
                phi_x_t = self.phi_x(x_t)
                y_hat_phy = self.mephy(u[:,:,t], x_t)
                if y_hat_phy.dtype != self.y_phi_phy[0].weight.dtype:
                    y_hat_phy_f32 = y_hat_phy.to(self.y_phi_phy[0].weight.dtype)
                    y_phy_phi = self.y_phi_phy(y_hat_phy_f32)
                else:
                    y_phy_phi = self.y_phi_phy(y_hat_phy)
                    
                y_hat = self.menn_phy(torch.cat([phi_x_t,y_phy_phi],1))
                loss += torch.sum((y_hat-y[:, :, t]) ** 2)
            elif self.mpnt_wt<=0:
                #pure nn
                phi_x_t = self.phi_x(x_t)
                # y_hat_phy = self.mephy(u[:,:,t], x_t)
                y_hat_nn = self.menn(phi_x_t)
                loss += torch.sum((y_hat_nn-y[:, :, t]) ** 2)
                
            else:               
                ## add measurement known panalty
                phi_x = self.phi_x(x_t)

                y_hat = self.menn(phi_x)
                measure_mean_t, measure_var_t = self.mephy(x_mean, x_logvar )
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
            # torch.autograd.set_detect_anomaly(True)
            if t == 0:
                x_tm1 =  torch.zeros(batch_size, self.z_dim, device=self.device)
            else: 
                x_tm1 = x[:,:,t-1]

            # feature extraction: u_t
            phi_u_t = self.phi_u(u[:, :, t])

            if self.mpnt_wt>100:
                # pure physical
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1)
                x_t = x_mean_phy
            elif  self.mpnt_wt>=10:
                # physical augmentation CX
                dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
                x_mean_nn = self.x_mean(dynn_phi)

                x_mean_phy = self.dyphy(u[:, :, t],x_tm1)
                x_t = x_mean_nn + x_mean_phy
                
            elif self.mpnt_wt<=-10:
                #physics guided
                x_mean_phy = self.dyphy(u[:, :, t],x_tm1)
                if x_mean_phy.dtype != self.x_phi_phy[0].weight.dtype:
                    x_mean_phy_f32 = x_mean_phy.to(self.x_phi_phy[0].weight.dtype)
                    x_phy_phi = self.x_phi_phy(x_mean_phy_f32)
                else:
                    x_phy_phi = self.x_phi_phy(x_mean_phy)
                dynn_phi = self.dynn_phy(torch.cat([phi_u_t, h[-1],x_phy_phi], 1))
                x_mean_nn = self.x_mean(dynn_phi)
                x_t = x_mean_nn
                
            elif self.mpnt_wt<=0:
                #pure nn
                dynn_phi = self.dynn(torch.cat([phi_u_t, h[-1]], 1))
                x_mean_nn = self.x_mean(dynn_phi)
                x_t = x_mean_nn

            # phi_x = self.phi_x(x[:,:,t])
            # y_hat[:, :, t] = self.menn(phi_x)
            
            x[:,:,t] = x_t 
            # recurrence: u_t+1 -> h_t+1
            _, h = self.rnn(phi_u_t.unsqueeze(0), h)
                        # print("all_loss", loss)
                        
            if self.mpnt_wt>100:
                # pure physical constraints
                y_hat_phy = self.mephy(u[:,:,t],x_t)
                y_hat[:, :, t] = y_hat_phy
                
                        
            elif self.mpnt_wt>=10:
                
                # physical augmentation CX
                phi_x_t = self.phi_x(x_t)
                y_hat_nn = self.menn(phi_x_t)
                y_hat_phy = self.mephy(u[:,:,t],x_t)
                y_hat[:, :, t] = y_hat_nn+y_hat_phy
            elif self.mpnt_wt<=-10:
                # physics guided 
                phi_x_t = self.phi_x(x_t)
                y_hat_phy = self.mephy(u[:,:,t], x_t)
                y_phy_phi = self.y_phi_phy(y_hat_phy)
                y_hat[:, :, t] = self.menn_phy(torch.cat([phi_x_t,y_phy_phi],1))

            elif self.mpnt_wt<=0:
                # pure nn
                # remember to add y_var (maybe not)
                phi_x = self.phi_x(x_mean_nn)
                y_hat[:, :, t] = self.menn(phi_x)
                
            else:
                
                # panelty
                phi_x = self.phi_x(torch.cat([x[:,:,t], x_logvar], 1))
                y_hat[:, :, t]  = self.menn(phi_x)

         
        y_hat_mu = y_hat

        return y_hat, y_hat_mu, y_hat_sigma, x

