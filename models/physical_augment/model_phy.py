from models.physical_augment.kuka300 import kuka300
from models.physical_augment.randomRobo import randomRobo
import roboticstoolbox as rtb


# import kuka300
# import toy_lgssm
import torch
import torch.nn as nn
import numpy as np

class MODEL_PHY():
    def __init__(self, phy_type, sysparam, device ):
        self.phy_type = phy_type
        self.param = sysparam
        self.device = device
        self.if_clip = self.param['if_clip']
        self.if_G = self.param['if_G']
        print("self.param['roboname']: ",self.param['roboname'])
        print("if_clip: ",self.if_clip)
        print("if_G: ",self.if_G)
        
        if self.phy_type == 'industrobo':
            if self.param['roboname'] == "KUKA300":
                self.model = kuka300(robot_type="correct")
                self.dof = self.model.dof
            elif self.param['roboname'] == "KUKA300noffset":
                self.model = kuka300(robot_type="no_offset")
                self.dof = self.model.dof
                
            elif self.param['roboname'] == "Puma560":
                # self.if_clip = False
                # self.if_G = False
                self.model = rtb.models.DH.Puma560()
                self.dof = 6
            elif self.param['roboname'] == "randomRobo":
                # self.if_clip = False
                # self.if_G = False
                self.model = randomRobo()
                self.dof = 6
                
        elif self.phy_type == 'toy_lgssm':
            # decide how to change or where to add the congifuration that what parts of the models are available (do I need seperated model for that or maybe, no)
            self.model == toy_lgssm()# can be initialed by adding A,B,C,D matrix here
    
    def qdd_func(self, t, q, qd):
        # Define your acceleration function here.
        # For example, let's assume a simple linear system qdd = -k * q - b * qd
        k = 1.0
        b = 0.1
        return -k * q - b * qd
            
    def rk4_step(self, t, q, qd, dt, qdd_prev):
        k1 = self.qdd_func(t, q, qd)
        k2 = self.qdd_func(t + 0.5 * dt, q + 0.5 * dt * qd, qd + 0.5 * dt * k1)
        k3 = self.qdd_func(t + 0.5 * dt, q + 0.5 * dt * (qd + 0.5 * dt * k1), qd + 0.5 * dt * k2)
        k4 = self.qdd_func(t + dt, q + dt * (qd + 0.5 * dt * k3), qd + dt * k3)
        
        qdd = (k1 + 2 * k2 + 2 * k3 + k4) / 6

        # Update position and velocity using the kinematic equations
        q_new = q + qd * dt + 0.5 * qdd_prev * (dt ** 2) + (1/6) * (qdd - qdd_prev) * (dt ** 2)
        qd_new = qd + 0.5 * (qdd + qdd_prev) * dt
        
        return q_new, qd_new, qdd
       
    def dynamic_model(self, u, x_pre, if_clip_qd = True, if_clip_q=True):
        if_clip_qd = self.if_clip
        if_clip_q = self.if_clip
        
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
            dof_num = self.dof


            q_lim_max =[]
            q_lim_min = []
            # print("torque[0] before G",torque[0]) 

            for i in range(dof_num):
                # Update torque by multiplying by gear ratio (assuming self.model.links[i].G is a scalar)
                torch.autograd.set_detect_anomaly(True)
                if self.if_G == True:
                    G = [212.76, 203.52, 192.75, 156, 156, 102.17]
                    torque[:, i] = u[:, i]*G[i]
                
                # Append joint limits to lists
                if self.if_clip == True: 
                    q_min = [-90*np.pi/180, -30*np.pi/180, -110*np.pi/180, -180*np.pi/180, -90*np.pi/180, -180*np.pi/180]      
                    q_max = [90*np.pi/180,  40*np.pi/180,  40*np.pi/180,  180*np.pi/180, 90*np.pi/180,  180*np.pi/180]
              
                    qd_lim = [63.4, 61.7, 59.5 , 91.5,85.8,131.3]
                    print("q_min set")
                else:
                    q_min = [-180*np.pi/180, -180*np.pi/180, -180*np.pi/180, -180*np.pi/180, -180*np.pi/180, -180*np.pi/180]      
                    q_max = [180*np.pi/180,  180*np.pi/180,  180*np.pi/180,  180*np.pi/180, 180*np.pi/180,  180*np.pi/180]
                    qd_lim = [360, 360, 360, 360, 360, 360]
                    print("q_min not set")
                    

            q_lim_min.append(torch.tensor(q_min[i]))
            q_lim_max.append(torch.tensor(q_max[i]))
    
            q_lim_min = torch.hstack(q_lim_min).to(device='cuda')
            q_lim_max = torch.hstack(q_lim_max).to(device='cuda')
            # Integrate the equations of motion


            # Convert initial joint angles and velocities to tensors

            q_list = []
            qd_list = []
            qdd_list = []
            dt = self.param['dt']
            

            qd_lim_min = []
            qd_lim_max = []

            for i in qd_lim:
                qd_lim_min.append(-i/180*np.pi)
                qd_lim_max.append(i/180*np.pi)
            
            qd_lim_min = torch.tensor(qd_lim_min).to(device='cuda')
            qd_lim_max = torch.tensor(qd_lim_max).to(device='cuda')
            # pi = 3.1415926
            for i_batch in range(batch_size):
                # Compute joint accelerations using forward dynamics
                
                q_np = q[i_batch].clone().detach().cpu().numpy()
                qd_np = qd[i_batch].clone().detach().cpu().numpy()
                torque_np = torque[i_batch].clone().detach().cpu().numpy()
                
                
                qdd = self.model.accel(q_np, qd_np, torque_np, gravity = [0,0,9.81])
                q_i, qd_i, qdd_i = self.rk4_step(0, q_np, qd_np, dt, qdd)

                # Integrate to update joint velocities and positions
                qd_ib = torch.tensor(qd_i).to(device='cuda')
                q_ib = torch.tensor(q_i).to(device='cuda')
                
                if self.if_clip == True:
                    # qd_i= np.clip(qd_i,qd_lim_min,qd_lim_max )
                    qd_ib = torch.max(torch.min(qd_ib, qd_lim_max), qd_lim_min)
                    q_ib = torch.max(torch.min(q_ib, q_lim_max), q_lim_min)
                    
                
                
                # qd_ib = torch.tensor(qd_i).to(device='cuda')
                # q_ib = torch.tensor(q_i).to(device='cuda')
   
                
                # q_ib = torch.max(torch.min(q_ib, q_lim_max), q_lim_min)
    
                # Append results to lists
                q_list.append(q_ib)
                qd_list.append(qd_ib)
                qdd_list.append(torch.tensor(qdd).to(device='cuda'))

            q_array = torch.stack(q_list)
            qd_array = torch.stack(qd_list)
            qdd_array = torch.stack(qdd_list)

            xt =  torch.cat((q_array,qd_array), dim=1).to(dtype=torch.float32,device='cuda')
            # print(q_ib.device,q_array.device,xt.device)
        elif self.phy_type == 'toy_lgssm' or self.phy_type == 'toy_lgssm_5_pre':
            batch_size = u.shape[0]
            
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
            
            '''
            yt = xt[:,0:int(x_dim/2)].clone()
            instead of
            yt = xt[:,0:int(x_dim/2)] 
            to avoid inplace problem
            https://discuss.pytorch.org/t/runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation-torch-floattensor-64-1-which-is-output-0-of-asstridedbackward0-is-at-version-3-expected-version-2-instead-hint-the-backtrace-further-a/171826/7
            '''
            return yt
        
        elif self.phy_type == 'toy_lgssm' or self.phy_type == 'toy_lgssm_5_pre':
            batch_size = ut.shape[0]
            C_prt = torch.tensor( self.param['C_prt'], dtype=torch.float32,device=self.device)
            C_prt = C_prt.expand(batch_size, -1, -1)
            # print(C.shape,y.shape,ut.shape)
            y = torch.matmul(C_prt,xt.unsqueeze(-1)).squeeze(-1)
            return y
        #         def measure_phy(self, z_mean_t, z_logvar_t):
        # if "lgssm" in self.dataset:
            # z_mean_t, z_logvar_t = xt
            # C = torch.tensor( self.param['C'], dtype=torch.float32,device=self.device)
            # sigma = torch.tensor( self.param['sigma_out'], dtype=torch.float32,device=self.device)
            # sigma2 = torch.pow(sigma,2)
            # z_logvar_t = z_logvar_t.unsqueeze(-1)
            # z_var_t = torch.matmul(C*(z_logvar_t.exp()),C.T)
            # measure_mean = torch.matmul(C,z_mean_t.unsqueeze(-1)).squeeze(-1)
            # measure_var = (z_var_t+sigma2).squeeze(-1)                     
            # return (measure_mean, measure_var)
    
            
