from models.physical_augment.kuka300 import kuka300
# import kuka300
# import toy_lgssm
import torch
import torch.nn as nn
import numpy as np

class MODEL_PHY():
    def __init__(self, phy_type):
        self.phy_type = phy_type
        if self.phy_type == 'industrobo':
            self.model = kuka300()
        elif self.phy_type == 'toy_lgssm':
            # decide how to change or where to add the congifuration that what parts of the models are available (do I need seperated model for that or maybe, no)
            self.model == toy_lgssm()# can be initialed by adding A,B,C,D matrix here
            
    def dynamic_model(self, u, x_pre):
        if self.phy_type == 'industrobo':
            # forward dynamics (start from 6 dim)
            # Initial conditions
            batch_size = u.shape[0]
            x_dim = x_pre.shape[1]
            dof = int(x_dim/2)

            q = x_pre[:,0:dof].clone()  # Initial joint positions
            qd = x_pre[:,dof:].clone() # Initial joint velocities

            # q = torch.tensor(x_pre[:,0:dof])  
            # qd = torch.tensor(x_pre[:,dof:])  

            torque = torch.zeros(u.size())
            
            
            # torque = u[:,1:dof]  # Example constant torques
            # time_test = u[:,0]
            dof_num = self.model.dof


            q_lim_max =[]
            q_lim_min = []
            # print("torque[0] before G",torque[0]) 

            for i in range(dof_num):
                # Update torque by multiplying by gear ratio (assuming self.model.links[i].G is a scalar)
                torch.autograd.set_detect_anomaly(True)

                torque[:, i] = u[:, i]*self.model.links[i].G
                
                # Append joint limits to lists
                q_lim_min.append(torch.tensor(self.model.links[i].qlim[0]))
                q_lim_max.append(torch.tensor(self.model.links[i].qlim[1]))
            # print("torque[0] after G",torque[0]) 
            q_lim_min = torch.hstack(q_lim_min).to(device='cuda')
            q_lim_max = torch.hstack(q_lim_max).to(device='cuda')
            # Integrate the equations of motion


            # Convert initial joint angles and velocities to tensors

            q_list = []
            qd_list = []
            qdd_list = []
            dt = 1
            # pi = 3.1415926
            for i_batch in range(batch_size):
                # Compute joint accelerations using forward dynamics
                # print("q[i_batch], qd[i_batch], torque[i_batch]",q[i_batch], qd[i_batch], torque[i_batch])
                # print("q[i_batch].shape, qd[i_batch].shape, torque[i_batch].shape",q[i_batch].shape, qd[i_batch].shape, torque[i_batch].shape)
                
                # qdd = self.model.accel(np.array(q[i_batch]), np.array(qd[i_batch]), np.array(torque[i_batch]))
                # print(q[i_batch].device)
                
                q_np = q[i_batch].clone().detach().cpu().numpy()
                qd_np = qd[i_batch].clone().detach().cpu().numpy()
                torque_np = torque[i_batch].clone().detach().cpu().numpy()
                
                
                qdd = self.model.accel(q_np, qd_np, torque_np)
                

                # Integrate to update joint velocities and positions
                qd_ib = qd[i_batch] + torch.tensor(qdd * dt).to(device='cuda')
                # print(qd_ib.device,q[i_batch].device)
                q_ib = q[i_batch] + qd_ib * torch.tensor(dt).to(device='cuda')
                q_ib = torch.max(torch.min(q_ib, q_lim_max), q_lim_min)
                qd_ib 
                # Append results to lists
                q_list.append(q_ib)
                qd_list.append(q_ib)
                qdd_list.append(torch.tensor(qdd))

            q_array = torch.stack(q_list)
            qd_array = torch.stack(qd_list)
            qdd_array = torch.stack(qdd_list)

            xt =  torch.cat((q_array,qd_array), dim=1).to(device='cuda')
            # print(q_ib.device,q_array.device,xt.device)
        elif self.phy_type == 'toy_lgssm':

            A_prt =torch.tensor(self.param['A_prt'], dtype=torch.float32,device=self.device)
            A_prt = A_prt.expand(batch_size, -1, -1)

            B_prt =torch.tensor(self.param['B_prt'], dtype=torch.float32,device=self.device)
            B_prt = B_prt.expand(batch_size, -1, -1)
            xt =  torch.matmul(A_prt,x_pre.unsqueeze(-1)).squeeze(-1)+torch.matmul(B_prt,u.unsqueeze(-1)).squeeze(-1)

        return xt
    
    def measurement_model(self, ut, xt):
        if self.phy_type == 'industrobo':
            # the output the position which is the first dim of state 
            # st.shape (batch_size, number of state (each state is 6 dim!))
            x_dim = xt.shape[1]
            yt = xt[:,0:int(x_dim/2)].clone()
            return yt
        elif self.phy_type == 'toy_lgssm':
        #         def measure_phy(self, z_mean_t, z_logvar_t):
        # if "lgssm" in self.dataset:
            z_mean_t, z_logvar_t = xt
            C = torch.tensor( self.param['C'], dtype=torch.float32,device=self.device)
            sigma = torch.tensor( self.param['sigma_out'], dtype=torch.float32,device=self.device)
            sigma2 = torch.pow(sigma,2)
            z_logvar_t = z_logvar_t.unsqueeze(-1)
            z_var_t = torch.matmul(C*(z_logvar_t.exp()),C.T)
            measure_mean = torch.matmul(C,z_mean_t.unsqueeze(-1)).squeeze(-1)
            measure_var = (z_var_t+sigma2).squeeze(-1)                     
            return (measure_mean, measure_var)
    
            # return y
