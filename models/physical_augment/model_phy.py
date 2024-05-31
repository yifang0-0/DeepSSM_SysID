import kuka300
import toy_lgssm
import torch
import torch.nn as nn
import numpy as np

class MODEL_PHY():
    def __init__(self, phy_type):
        self.phy_type = phy_type
        if self.phy_type == 'kuka300':
            self.model = kuka300()
        elif self.phy_type == 'toy_lgssm':
            # decide how to change or where to add the congifuration that what parts of the models are available (do I need seperated model for that or maybe, no)
            self.model == toy_lgssm()# can be initialed by adding A,B,C,D matrix here
            
    def dynamic_model(self, u, x_pre, is_init, options):
        if self.phy_type == 'kuka300':
            # forward dynamics (start from 6 dim)
            # Initial conditions
            batch_size = u.shape[0]
            if is_init:
                q0 = torch.tensor(self.model.qz)   # Initial joint positions
                qd0 = torch.tensor(self.model.qd0)  # Initial joint velocities
            else:
                q0 = torch.tensor(self.model.qz)   # Initial joint positions
                qd0 = torch.tensor(self.model.qd0)  # Initial joint velocities
                
            torque = torch.zeros((batch_size,6) + u.shape[2:])
            time_test = torch.zeros((batch_size,1) + u.shape[2:])
            
            torque = u[:,1:7]  # Example constant torques
            time_test = u[:,0]
            dof_num = self.model.dof


            q_lim_max =[]
            q_lim_min = []

            for i in range(dof_num):
                # Update torque by multiplying by gear ratio (assuming self.model.links[i].G is a scalar)
                torque[:, i] *= self.model.links[i].G
                
                # Append joint limits to lists
                q_lim_min.append(torch.tensor(self.model.links[i].qlim[0]))
                q_lim_max.append(torch.tensor(self.model.links[i].qlim[1]))
                
            
                # Integrate the equations of motion
                q_list = []
                qd_list = []
                qdd_list = []

                # Convert initial joint angles and velocities to tensors
                q = torch.tensor(q0)
                qd = torch.tensor(qd0)

            for index,dt in enumerate(torch.diff(time_test,  dim=0)):
                # Ensure joint positions are within limits
                q = torch.clamp(q, q_lim_min, q_lim_max)
                
                # Compute joint accelerations using forward dynamics
                qdd = self.model.accel(q, qd, torque[index])
                
                # Integrate to update joint velocities and positions
                q = q + qd * dt
                qd = qd + qdd * dt
                
                # Append results to lists
                q_list.append(q)
                qd_list.append(qd)
                qdd_list.append(qdd)

            q_array = torch.stack(q_list)
            qd_array = torch.stack(qd_list)
            qdd_array = torch.stack(qdd_list)
            
            xt = torch.stack(q_array,qd_array)
            
        elif self.phy_type == 'toy_lgssm':

            A_prt =torch.tensor(self.param['A_prt'], dtype=torch.float32,device=self.device)
            A_prt = A_prt.expand(batch_size, -1, -1)

            B_prt =torch.tensor(self.param['B_prt'], dtype=torch.float32,device=self.device)
            B_prt = B_prt.expand(batch_size, -1, -1)
            xt =  torch.matmul(A_prt,x_pre.unsqueeze(-1)).squeeze(-1)+torch.matmul(B_prt,u.unsqueeze(-1)).squeeze(-1)

        return xt
    
    def measurement_model(self, ut, xt, options):
        if self.phy_type == 'kuka300':
            # the output the position which is the first dim of state 
            # st.shape (batch_size, number of state (each state is 6 dim!))
            return xt[:,0:6]
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
