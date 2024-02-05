# import generic libraries
"""
The `run_main_single` function is the main function that runs the training and testing of a single
model on a given dataset.

:param options: - dataset: The name of the dataset to use. Options are 'narendra_li', 'toy_lgssm',
'wiener_hammerstein'
:param path_general: The `path_general` variable is the path where the log files and data files will
be saved. It is a combination of the current working directory, the log directory specified in the
options, the dataset name, and the model name
:param file_name_general: The `file_name_general` parameter is a string that specifies the general
file name for saving the results of the experiment. It is used to create a unique file name for each
experiment by appending additional information such as the model type, dynamic system type, and
model parameters (h_dim, z_dim,
"""

import torch.utils.data
import pandas as pd
import os
import torch
import time
import sys
import matplotlib.pyplot as plt
import argparse
import subprocess
import io
# os.chdir('../')
# sys.path.append(os.getcwd())
# import user-written files
import data.loader as loader
import training
import testing
from utils.utils import compute_normalizer
from utils.logger import set_redirects
from utils.utils import save_options
# import options files
import options.train_options as main_params
import options.train_options as model_params
import options.train_options as dynsys_params
import options.train_options as train_params
from models.model_state import ModelState


# %%####################################################################################################################
# Main function
########################################################################################################################
def run_main_single(options, path_general, file_name_general):
    start_time = time.time()
    print(time.strftime("%c"))
    
   # get correct computing device
    if torch.cuda.is_available():
        
        # get the usage of gpu memory from command "nvidia-smi" 
        gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
        gpu_df = pd.read_csv(io.BytesIO(gpu_stats),names=['memory.used', 'memory.free'],skiprows=1)
        print('GPU usage:\n{}'.format(gpu_df))
        
        #get the id of the gpu with a maximum memory space left
        gpu_df['memory.free'] = gpu_df['memory.free'].map(lambda x: x.rstrip(' [MiB]'))
        idx = gpu_df['memory.free'].astype(int).idxmax()        
        print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
        
        # run the task on the selected GPU
        torch.cuda.set_device(idx)
        device = torch.device('cuda') 
        gpu_name = torch.cuda.get_device_name(idx)
        print(f"Using GPU {idx}: {gpu_name}") 
    else:
        device = torch.device('cpu')
    print('Device: {}'.format(device))


    # get the options
    options['device'] = device
    options['dataset_options'] = dynsys_params.get_dataset_options(options['dataset'])
    options['model_options'] = model_params.get_model_options(options['model'], options['dataset'],
                                                              options['dataset_options'])
    options['train_options'] = train_params.get_train_options(options['dataset'])
    options['test_options'] = train_params.get_test_options()

    # print model type and dynamic system type
    print('\n\tModel Type: {}'.format(options['model']))
    print('\tDynamic System: {}\n'.format(options['dataset']))

    file_name_general = file_name_general + '_h{}_z{}_n{}'.format(options['model_options'].h_dim,
                                                                  options['model_options'].z_dim,
                                                                  options['model_options'].n_layers)
    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)
    # set logger
    set_redirects(path, file_name_general)

    # Specifying datasets
    loaders = loader.load_dataset(dataset=options["dataset"],
                                  dataset_options=options["dataset_options"],
                                  train_batch_size=options["train_options"].batch_size,
                                  test_batch_size=options["test_options"].batch_size, 
                                  known_parameter=options["known_parameter"])

    # Compute normalizers
    if options["normalize"]:
        normalizer_input, normalizer_output = compute_normalizer(loaders['train'])
    else:
        normalizer_input = normalizer_output = None

    if options['known_parameter'] == 'B':
        options['model_options'].u_dim = 2

        
    # Define model
    modelstate = ModelState(seed=options["seed"],
                            nu=loaders["train"].nu, ny=loaders["train"].ny,
                            model=options["model"],
                            options=options,
                            normalizer_input=normalizer_input,
                            normalizer_output=normalizer_output)
    modelstate.model.to(options['device'])

    # save the options
    save_options(options, path_general, 'options.txt')

    # correct input size

        

    # allocation
    
    
    if options['do_train']:
        df = {}
        # train the model
        df = training.run_train(modelstate=modelstate,
                                loader_train=loaders['train'],
                                loader_valid=loaders['valid'],
                                options=options,
                                dataframe=df,
                                path_general=path_general,
                                file_name_general=file_name_general)
        df = pd.DataFrame(df)

    if options['do_test']:
        # # test the model
        # df = testing.run_test(options, loaders, df, path_general, file_name_general)
        # df = pd.DataFrame(df)
        
        
        # test the model for 10 times
        # make sure df is in dataframe format
        df = pd.DataFrame({})
        for i in range(10):
            df_single = {}
            # test the model
            df_single = testing.run_test(options, loaders, df_single, path_general, file_name_general)
            # make df_single a dataframe
            df_single = pd.DataFrame(df_single)
            df = pd.concat([df,df_single])
            
        
    # %% save data
    # get saving path
    path = path_general + 'data/'
    # check if path exists and create otherwise
    if not os.path.exists(path):
        os.makedirs(path)


    # filename
    file_name = file_name_general + '.csv'
    
    # %% apend on the old one check if there's old log
    # if os.path.exists(path + file_name):
    #     # read old data
    #     df_old = pd.read_csv(path + file_name,index_col=None)
    #     #append new to the old file
    #     df = pd.concat([df_old,df])
    
    
    # save data
    if options['savelog']:
        df.to_csv(path + file_name,index=False)


    # time output
    time_el = time.time() - start_time
    hours = time_el // 3600
    min = time_el // 60 - hours * 60
    sec = time_el - min * 60 - hours * 3600
    print('Total ime of file execution: {}:{:2.0f}:{:2.0f} [h:min:sec]'.format(hours, min, sec))
    print(time.strftime("%c"))


# %%
# The `if __name__ == "__main__":` block is used to check if the current script is being run as the
# main program. If it is, then the code inside the block will be executed.

if __name__ == "__main__":
    # set (high level) options dictionary, if the basic options are expected from the augment parser, we set OPTION_SETTING_MANUALLY = True, else we change the options directly from the python file.
    OPTION_FROM_PARSER = False
    if OPTION_FROM_PARSER is True:
        options = {}

        main_params_parser = main_params.get_main_options()
        options['dataset'] = main_params_parser.dataset
        options['model'] = main_params_parser.model
        options['do_train'] = main_params_parser.do_train
        options['do_test'] = main_params_parser.do_test
        options['logdir'] = main_params_parser.logdir
        options['normalize'] = main_params_parser.normalize
        options['seed'] = main_params_parser.seed
        options['optim'] = main_params_parser.optim
        options['showfig'] = main_params_parser.showfig
        options['savefig'] = main_params_parser.savefig
        options['known_parameter'] = main_params_parser.known_parameter

        # print("Encountered errors loading the main options of the training/testing task")
        
        
    else:
        options = {
            'dataset': 'toy_lgssm',  # options: 'narendra_li', 'toy_lgssm', 'wiener_hammerstein'
            'model': 'VAE-RNN-PHY', # options: 'VAE-RNN', 'VRNN-Gauss', 'VRNN-Gauss-I', 'VRNN-GMM', 'VRNN-GMM-I', 'STORN'
            'do_train': True,
            'do_test': True,
            'logdir': 'single_phy_idmaC',
            'normalize': True,
            'seed': 1234,
            'optim': 'Adam',
            'showfig': False,
            'savefig': True,
            'savelog': False,
            'known_parameter': 'None'
        }

    # get saving path
    path_general = os.getcwd() + '/log/{}/{}/{}_{}/'.format(options['logdir'],
                                                         options['dataset'],
                                                         options['model'],options['known_parameter'] )

    # get saving file names
    file_name_general = options['dataset']

    run_main_single(options, path_general, file_name_general)
