import argparse
import torch
import numpy as np

'''
parse_known_args return the new parser and the already know data
'''

def get_train_options(dataset_name):
    train_parser = argparse.ArgumentParser(description='training parameter')
    train_parser.add_argument('--clip', type=int, default=10, help='clipping of gradients')
    train_parser.add_argument('--lr_scheduler_nstart', type=int, default=10, help='learning rate scheduler start epoch')
    train_parser.add_argument('--print_every', type=int, default=1, help='output print of training')
    train_parser.add_argument('--test_every', type=int, default=5, help='test during training after every n epoch')
    
    if dataset_name == 'narendra_li':
        train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-6, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=10, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')

    elif dataset_name == 'toy_lgssm':
        train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-7, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=30, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=5, help='adapt learning rate by')
        # train_parser.add_argument('--unknown_parameter', type=int, default=1, help='0 is the normal training, 1 means linear matrix B is known')
    
    elif dataset_name == 'toy_lgssm_5_pre':
        train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-7, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=30, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=5, help='adapt learning rate by')
        
    elif dataset_name == 'toy_lgssm_2dy_5_pre':
        train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-7, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=30, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=5, help='adapt learning rate by')


    elif dataset_name == 'wiener_hammerstein':
        train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-6, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=20, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')
        
    elif dataset_name == 'f16gvt':
        train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-6, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=20, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')
        
    elif dataset_name == 'industrobo':
        train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-6, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=20, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')

    # change batch size to higher value if trained on cuda device
    if torch.cuda.is_available():
        train_parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    else:
        train_parser.add_argument('--batch_size', type=int, default=128, help='batch size')


    train_options, unknown = train_parser.parse_known_args()

    return train_options


def get_test_options():
    test_parser = argparse.ArgumentParser(description='testing parameter')
    test_parser.add_argument('--batch_size', type=int, default=32, help='batch size')  # 128
    test_options,unkonwn = test_parser.parse_known_args()

    return test_options


def get_main_options():

    # model parameters
    model_parser = argparse.ArgumentParser(description='Model Parameter')
    model_parser.add_argument('--dataset',metavar='', type=str, default='toy_lgssm')
    model_parser.add_argument('--model', metavar = '', type=str, default='VRNN-Gauss-I')
    model_parser.add_argument('--do_train', action="store_true")
    model_parser.add_argument('--do_test', action="store_true")
    model_parser.add_argument('--logdir',metavar = '',  type=str, default='same_dataset')
    model_parser.add_argument('--normalize', metavar = '', type=bool, default=True)
    model_parser.add_argument('--seed', metavar = '', type=int, default=1234)
    model_parser.add_argument('--optim', metavar = '', type=str, default='Adam')
    model_parser.add_argument('--showfig', metavar = '', type=bool, default=True)
    model_parser.add_argument('--savefig', metavar = '', type=bool, default=True)
    model_parser.add_argument('--savelog', metavar = '', type=bool, default=True)
    model_parser.add_argument('--saveoutput', metavar = '', type=bool, default=True)
    model_parser.add_argument('--known_parameter', metavar = '', type=str, default='None')
    model_parser.add_argument('--train_rounds', metavar = '', type=int, default=1)
    model_parser.add_argument('--start_from', metavar = '', type=int, default=0)
    
    model_options, unknown = model_parser.parse_known_args()
    print(model_options)
    return model_options


def get_system_options(dataset_name,dataset_options):
    if dataset_name == 'toy_lgssm_5_pre' or dataset_name == 'toy_lgssm':
        lgssm_system_parameter = {}
        lgssm_system_parameter['A'] = np.array([[0.7, 0.8], [0, 0.1]])
        lgssm_system_parameter['B'] = np.array([[-1], [0.1]])
        if dataset_options.A_prt_idx==0:
            lgssm_system_parameter['A_prt'] = np.array([[0, 0], [0, 0]]) 
        elif dataset_options.A_prt_idx==1:
            lgssm_system_parameter['A_prt'] = np.array([[0.6, 0.7], [-0.1, 0]])
        elif dataset_options.A_prt_idx==2:
            lgssm_system_parameter['A_prt'] = np.array([[0.7, 0.8], [0, 0.1]])  
            
        if dataset_options.B_prt_idx==0:
            lgssm_system_parameter['B_prt'] = np.array([[0], [0]]) 
        elif dataset_options.B_prt_idx==1:
            lgssm_system_parameter['B_prt'] = np.array([[-1.1], [0]])
        elif dataset_options.B_prt_idx==2:
            lgssm_system_parameter['B_prt'] = np.array([[-1], [0.1]])
    
        if dataset_options.C_prt_idx==0:
            lgssm_system_parameter['C_prt'] = np.array([[0, 0]]) 
        elif dataset_options.C_prt_idx==1:
            lgssm_system_parameter['C_prt'] = np.array([[0.9, -0.1]]) 
        elif dataset_options.C_prt_idx==2:
            lgssm_system_parameter['C_prt'] = np.array([[1, 0]]) 
        # lgssm_system_parameter['C'] = np.array([[1, 0]])
        lgssm_system_parameter['sigma_state'] = np.sqrt(0.25)
        lgssm_system_parameter['sigma_out'] = np.sqrt(1)
    
    elif dataset_name == 'toy_lgssm_2dy_5_pre':
        lgssm_system_parameter = {}
        lgssm_system_parameter['A'] = np.array([[0.7, 0.8], [0, 0.1]])
        lgssm_system_parameter['B'] = np.array([[-1], [0.1]])
        lgssm_system_parameter['C'] = np.array([[1, 0], [0, 1]]).transpose()
        lgssm_system_parameter['sigma_state'] = np.sqrt(0.25)
        lgssm_system_parameter['sigma_out'] = np.sqrt(1)
    else:
        lgssm_system_parameter = {}
    return lgssm_system_parameter

def get_dataset_options(dataset_name):

    """Not used datasets"""
    """if dataset_name == 'cascaded_tank':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: cascaded tank')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=128, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=128, help='validation sequence length')
        dataset_options = dataset_parser.parse_args()

    elif dataset_name == 'f16gvt':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: f-16')
        dataset_parser.add_argument('--y_dim', type=int, default=3, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=2048, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=2048, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=2048, help='validation sequence length')
        dataset_options = dataset_parser.parse_args()"""

    if dataset_name == 'narendra_li':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: narendra li')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=2000, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=2000, help='validation sequence length')  # 512
        dataset_options, unknown = dataset_parser.parse_known_args()

    elif dataset_name == 'toy_lgssm':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: lgssm')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=64, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--loss_type', type=int, default=0, help='0:normal loss, 1:measurement penalty')
        dataset_parser.add_argument('--A_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical A')
        dataset_parser.add_argument('--B_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical B')
        dataset_parser.add_argument('--C_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical C')
        
        dataset_parser.add_argument('--seq_len_val', type=int, default=64, help='validation sequence length')  # 512
        dataset_options, unknown = dataset_parser.parse_known_args()

    elif dataset_name == 'toy_lgssm_5_pre':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: lgssm')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=64, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=64, help='validation sequence length')  # 512
        dataset_parser.add_argument('--loss_type', type=int, default=0, help='0:normal loss, 1:measurement penalty')
        dataset_parser.add_argument('--A_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical A')
        dataset_parser.add_argument('--B_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical B')
        dataset_parser.add_argument('--C_prt_idx', type=int, default=0, help='0:no knowledge, 1:know with bias, 2:the identical C')
        dataset_parser.add_argument('--k_max_train', type=int, default=2000, help='training set length')
        dataset_parser.add_argument('--k_max_test', type=int, default=5000, help='test set length')
        dataset_parser.add_argument('--k_max_val', type=int, default=2000, help='validation set length')  # 512
        dataset_options, unknown = dataset_parser.parse_known_args()
    
    elif dataset_name == 'toy_lgssm_2dy_5_pre':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: lgssm')
        dataset_parser.add_argument('--y_dim', type=int, default=2, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=64, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=64, help='validation sequence length')  # 512
        dataset_parser.add_argument('--loss_type', type=int, default=0, help='0:normal loss, 1:measurement penalty')
        dataset_parser.add_argument('--k_max_train', type=int, default=2000, help='training set length')
        dataset_parser.add_argument('--k_max_test', type=int, default=5000, help='test set length')
        dataset_parser.add_argument('--k_max_val', type=int, default=2000, help='validation set length')  # 512
        dataset_options, unknown = dataset_parser.parse_known_args()


    elif dataset_name == 'wiener_hammerstein':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: wiener hammerstein')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=2048, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=2048, help='validation sequence length')
        dataset_options, unknown = dataset_parser.parse_known_args()
        
    elif dataset_name == 'f16gvt':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: f-16')
        dataset_parser.add_argument('--y_dim', type=int, default=3, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=2, help='dimension of u')
        dataset_parser.add_argument('--input_lev', type=int, default=7, help='input activation level')
        dataset_parser.add_argument('--input_type', type=str, default="FullMSine", help='input activation level')
        dataset_parser.add_argument('--seq_len_train', type=int, default=1024, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=1024, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=1024, help='validation sequence length')
        dataset_options, unknown = dataset_parser.parse_known_args()
        
    elif dataset_name == 'industrobo':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: industrobo')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=2, help='dimension of u, the input with time')
        dataset_parser.add_argument('--input_channel', type=int, default=1, help='train which joint')
        # dataset_parser.add_argument('--input_type', type=str, default="FullMSine", help='input activation level')
        dataset_parser.add_argument('--seq_stride', type=int, default=None, help='window size stride')
        dataset_parser.add_argument('--seq_len_train', type=int, default=256, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=64, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=64, help='validation sequence length')
        dataset_options, unknown = dataset_parser.parse_known_args()


    return dataset_options


def get_model_options(model_type, dataset_name, dataset_options):

    y_dim = dataset_options.y_dim
    u_dim = dataset_options.u_dim

    # model parameters
    model_parser = argparse.ArgumentParser(description='Model Parameter')
    model_parser.add_argument('--y_dim', type=int, default=y_dim, help='dimension of y')
    model_parser.add_argument('--u_dim', type=int, default=u_dim, help='dimension of u')

    """Not used datasets"""
    """if dataset_name == 'cascaded_tank':
        model_parser.add_argument('--h_dim', type=int, default=60, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable') 
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')  
        
    elif dataset_name == 'f16gvt':
        model_parser.add_argument('--h_dim', type=int, default=40, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')"""

    if dataset_name == 'narendra_li':
        model_parser.add_argument('--h_dim', type=int, default=60, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=10, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')
        
    elif dataset_name == 'toy_lgssm_5_pre':
        model_parser.add_argument('--h_dim', type=int, default=10, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')
        
    elif dataset_name == 'toy_lgssm_2dy_5_pre':
        model_parser.add_argument('--h_dim', type=int, default=10, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')

    elif dataset_name == 'toy_lgssm':
        model_parser.add_argument('--h_dim', type=int, default=70, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')

    elif dataset_name == 'wiener_hammerstein':
        model_parser.add_argument('--h_dim', type=int, default=50, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=3, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=3, help='number of RNN layers (GRU)')
    
    elif dataset_name == 'f16gvt':
        model_parser.add_argument('--h_dim', type=int, default=30, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=5, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=3, help='number of RNN layers (GRU)')
    
    elif dataset_name == 'industrobo':
        model_parser.add_argument('--h_dim', type=int, default=50, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=5, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=3, help='number of RNN layers (GRU)')

    # only if type is GMM
    if model_type == 'VRNN-GMM-I' or model_type == 'VRNN-GMM':
        model_parser.add_argument('--n_mixtures', type=int, default=5, help='number Gaussian output mixtures')

    model_parser.add_argument('--mpnt_wt', type=float, default=0, help='how heavy is the measurement matrix')

    model_options, unkown = model_parser.parse_known_args()

    return model_options