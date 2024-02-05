import argparse
import torch

'''
parse_known_args return the new parser and the already know data
'''

def get_train_options(dataset_name):
    train_parser = argparse.ArgumentParser(description='training parameter')
    train_parser.add_argument('--clip', type=int, default=10, help='clipping of gradients')
    train_parser.add_argument('--lr_scheduler_nstart', type=int, default=10, help='learning rate scheduler start epoch')
    train_parser.add_argument('--print_every', type=int, default=1, help='output print of training')
    train_parser.add_argument('--test_every', type=int, default=5, help='test during training after every n epoch')

    """Not used datasets"""
    """if dataset_name == 'cascaded_tank':
        train_parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-6, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=4, help='check learning rater after') # 50/10
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')

    elif dataset_name == 'f16gvt':
        train_parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-6, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=10/2, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')"""

    if dataset_name == 'narendra_li':
        train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-6, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=10, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')

    elif dataset_name == 'toy_lgssm':
        train_parser.add_argument('--n_epochs', type=int, default=750, help='number of epochs')
        train_parser.add_argument('--init_lr', type=float, default=1e-3, help='initial learning rate')
        train_parser.add_argument('--min_lr', type=float, default=1e-6, help='minimal learning rate')
        train_parser.add_argument('--lr_scheduler_nepochs', type=float, default=10, help='check learning rater after')
        train_parser.add_argument('--lr_scheduler_factor', type=float, default=10, help='adapt learning rate by')
        # train_parser.add_argument('--unknown_parameter', type=int, default=1, help='0 is the normal training, 1 means linear matrix B is known')

    elif dataset_name == 'wiener_hammerstein':
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
    model_parser.add_argument('--showfig', metavar = '', type=bool, default=False)
    model_parser.add_argument('--savefig', metavar = '', type=bool, default=False)
    model_parser.add_argument('--known_parameter', metavar = '', type=str, default='None')
    model_parser.add_argument('--train_rounds', metavar = '', type=int, default=50)
    model_parser.add_argument('--start_from', metavar = '', type=int, default=0)
    
    model_options, unknown = model_parser.parse_known_args()
    print(model_options)
    return model_options


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
        dataset_parser.add_argument('--seq_len_val', type=int, default=64, help='validation sequence length')  # 512
        dataset_options, unknown = dataset_parser.parse_known_args()

    elif dataset_name == 'wiener_hammerstein':
        dataset_parser = argparse.ArgumentParser(description='dynamic system parameter: wiener hammerstein')
        dataset_parser.add_argument('--y_dim', type=int, default=1, help='dimension of y')
        dataset_parser.add_argument('--u_dim', type=int, default=1, help='dimension of u')
        dataset_parser.add_argument('--seq_len_train', type=int, default=2048, help='training sequence length')
        dataset_parser.add_argument('--seq_len_test', type=int, default=None, help='test sequence length')
        dataset_parser.add_argument('--seq_len_val', type=int, default=2048, help='validation sequence length')
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

    elif dataset_name == 'toy_lgssm':
        model_parser.add_argument('--h_dim', type=int, default=70, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=2, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=1, help='number of RNN layers (GRU)')

    elif dataset_name == 'wiener_hammerstein':
        model_parser.add_argument('--h_dim', type=int, default=50, help='dimension of det. latent variable h')
        model_parser.add_argument('--z_dim', type=int, default=3, help='dimension of stoch. latent variable')
        model_parser.add_argument('--n_layers', type=int, default=3, help='number of RNN layers (GRU)')

    # only if type is GMM
    if model_type == 'VRNN-GMM-I' or model_type == 'VRNN-GMM':
        model_parser.add_argument('--n_mixtures', type=int, default=5, help='number Gaussian output mixtures')

    model_options, unkown = model_parser.parse_known_args()

    return model_options