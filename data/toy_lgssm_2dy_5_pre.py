import matplotlib.pyplot as plt
import numpy as np
from data.base import IODataset


def create_toy_lgssm_2dy_5_datasets(seq_len_train=None, seq_len_val=None, seq_len_test=None, **kwargs):

    # length of all data sets
    if bool(kwargs) and ("k_max_train" in kwargs):
        k_max_train = kwargs['k_max_train']
        k_max_val = kwargs['k_max_val']
        k_max_test = kwargs['k_max_test']
    else:
        # Default option
        k_max_train = 2000
        k_max_val = 2000
        k_max_test = 5000
    

    # test set input
    # file_path = 'data/Toy_LGSSM/toy_lgssm_testdata.npz'
    file_path_train = 'data/Toy_LGSSM/toy_lgssm_2dy_pre_trainingset_{}.npz'
    file_path_test = 'data/Toy_LGSSM/toy_lgssm_2dy_testing.npz'
    

    ith_lgssm = kwargs["ith_round"]
    
    test_data = np.load(file_path_test)
    u_test = test_data['u_test'][0:k_max_test]
    y_test = test_data['y_test'][0:k_max_test]
    
    train_data = np.load(file_path_train.format(ith_lgssm))
    u_train = train_data['u_train'][0:k_max_train]
    y_train = train_data['y_train'][0:k_max_train]
    u_val = train_data['u_val'][0:k_max_val]
    y_val = train_data['y_val'][0:k_max_val]
    
    print("dataset: ", file_path_train.format(ith_lgssm))

    dataset_train = IODataset(u_train, y_train, seq_len_train)
    dataset_val = IODataset(u_val, y_val, seq_len_val)
    dataset_test = IODataset(u_test, y_test, seq_len_test)

    return dataset_train, dataset_val, dataset_test
