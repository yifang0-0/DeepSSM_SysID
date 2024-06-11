import numpy as np
from data.base import IODataset
from scipy.io import loadmat

def create_industrobo_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None, seq_stride = None, input_type = "SineSw", input_lev = 3, **kwargs):

    file_name_train = "/home/ruiyuanli/dcscgpuserver1/DeepSSM_SysID/data/IndustRobo/forward_with_val_time.mat"
    # read the file into variable
    
    

    df_industRobo =  loadmat(file_name_train)

    ith_lgssm = kwargs["ith_round"]

    u_train = df_industRobo["u_train"]
    u_val   = df_industRobo["u_val"]
    u_test  = df_industRobo["u_test"]
    y_train = df_industRobo["y_train"]/180*np.pi
    y_val   = df_industRobo["y_val"]/180*np.pi
    y_test  = df_industRobo["y_test"]/180*np.pi

    # y_train = df_industRobo["y_train"]
    # y_val   = df_industRobo["y_val"]
    # y_test  = df_industRobo["y_test"]
    
    time_train  = df_industRobo["time_train"]
    time_val  = df_industRobo["time_val"]
    time_test  = df_industRobo["time_test"]
    # u_train = df_industRobo["u_train"][ith_lgssm-1,:]
    # y_train = df_industRobo["y_train"][ith_lgssm-1,:]
    # u_val   = df_industRobo["u_val"][ith_lgssm-1,:]
    # y_val   = df_industRobo["y_val"][ith_lgssm-1,:]
    # u_test  = df_industRobo["u_test"][ith_lgssm-1,:]
    # y_test  = df_industRobo["y_test"][ith_lgssm-1,:]
    

    
   
    # convert from list to numpy array
    
    u_train = np.asarray(u_train).T[::10]
    y_train = np.asarray(y_train).T[::10]
    u_val = np.asarray(u_val).T[::10]
    y_val = np.asarray(y_val).T[::10]
    u_test = np.asarray(u_test).T[::10]
    y_test = np.asarray(y_test).T[::10]
    
    # u_train = np.asarray(u_train).T
    # y_train = np.asarray(y_train).T
    # u_val = np.asarray(u_val).T
    # y_val = np.asarray(y_val).T
    # u_test = np.asarray(u_test).T
    # y_test = np.asarray(y_test).T
    # time_train  = np.asarray(time_train).T
    # time_val  = np.asarray(time_val).T
    # time_test  = np.asarray(time_test).T
    
    print("seq_stride: ", seq_stride)
    
    # %% maybe not including the specific time step yet
    # u_train = np.hstack((time_test, u_train))
    # u_val = np.hstack((time_test, u_val))
    # u_test = np.hstack((time_test, u_test))

    # since the dataset is small, a sliding window machenism is added
    dataset_train = IODataset(u_train, y_train, seq_len_train, seq_stride)
    dataset_val = IODataset(u_val, y_val, seq_len_val, seq_stride)
    dataset_test = IODataset(u_test, y_test, seq_len_test, None)

    print("dataset_train.u.shape, dataset_train.y.shape")
    print(dataset_train.u.shape, dataset_train.y.shape)
    print("dataset_val.u.shape, dataset_val.y.shape")
    print(dataset_val.u.shape, dataset_val.y.shape)
    print("dataset_test.u.shape, dataset_test.y.shape")
    print(dataset_test.u.shape, dataset_test.y.shape)
    return dataset_train, dataset_val, dataset_test
