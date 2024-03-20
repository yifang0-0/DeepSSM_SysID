# import generic libraries
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import user-writte files
import utils.datavisualizer as dv
import utils.dataevaluater as de
from utils.utils import get_n_params
from models.model_state import ModelState
from utils.utils import compute_normalizer
from utils.kalman_filter import run_kalman_filter

def run_test(options, loaders, df, path_general, file_name_general, **kwargs):
    # switch to cpu computations for testing
    # options['device'] = 'cpu'

    # %% load model

    # Compute normalizers (here just used for initialization, real values loaded below)
    if options["normalize"]:
        normalizer_input, normalizer_output = compute_normalizer(loaders['train'])
    else:
        normalizer_input = normalizer_output = None

    # Define model
    modelstate = ModelState(seed=options["seed"],
                            nu=loaders["train"].nu, ny=loaders["train"].ny,
                            model=options["model"],
                            options=options,
                            normalizer_input=normalizer_input,
                            normalizer_output=normalizer_output)
    modelstate.model.to(options['device'])

    # load model
    path = path_general + 'model/'
    file_name = file_name_general + '_bestModel.ckpt'
    modelstate.load_model(path, file_name)
    modelstate.model.to(options['device'])

    # %% plot and save the loss curve
    dv.plot_losscurve(df, options, path_general, file_name_general)

    # %% others

    if bool(kwargs):
        file_name_add = kwargs['file_name_add']
    else:
        # Default option
        file_name_add = ''
    file_name_general = file_name_add + file_name_general

    # get the number of model parameters
    num_model_param = get_n_params(modelstate.model)
    print('Model parameters: {}'.format(num_model_param))

    # %% RUN PERFORMANCE EVAL
    # %%

    # %% sample from the model
    for i, (u_test, y_test) in enumerate(loaders['test']):
        # getting output distribution parameter only implemented for selected models
        u_test = u_test.to(options['device'])
        
        # y_sample, y_sample_mu, y_sample_sigma, z_sample_mu, z_sample_sigma = modelstate.model.generate(u_test)
        # y_sample, y_sample_mu, y_sample_sigma= modelstate.model.generate(u_test)
        
        
        ## TODO: original code
        y_sample, y_sample_mu, y_sample_sigma, z = modelstate.model.generate(u_test)
        # convert to cpu and to numpy for evaluation
        # samples data
        y_sample_mu = y_sample_mu.cpu().detach().numpy()
        y_sample_sigma = y_sample_sigma.cpu().detach().numpy()
        # u_sample_mu = u_sample_mu.cpu().detach().numpy()
        # u_sample_sigma = u_sample_sigma.cpu().detach().numpy()
        
        # test data
        y_test = y_test.cpu().detach().numpy()
        y_sample = y_sample.cpu().detach().numpy()
        u_test = u_test.cpu().detach().numpy()
        z = z.cpu().detach().numpy()
    # get noisy test data for narendra_li
    if options['dataset'] == 'narendra_li':
        # original test set is unnoisy -> get noisy test set
        yshape = y_test.shape
        y_test_noisy = y_test + np.sqrt(0.1) * np.random.randn(yshape[0], yshape[1], yshape[2])
    elif options['dataset'] == 'toy_lgssm':
        # original test set is unnoisy -> get noisy test set
        yshape = y_test.shape
        y_test_noisy = y_test + np.sqrt(1) * np.random.randn(yshape[0], yshape[1], yshape[2])
    else:
        y_test_noisy = y_test

    # %% plot resulting predictions
    if options['dataset'] == 'narendra_li':
        # for narendra_li problem show test data mean pm 3sigma as well
        data_y_true = [y_test, np.sqrt(0.1) * np.ones_like(y_test)]
        data_y_sample = [y_sample_mu, y_sample_sigma]
        label_y = ['true, $\mu\pm3\sigma$', 'sample, $\mu\pm3\sigma$']
    elif options['dataset'] == 'toy_lgssm':
        # for lgssm problem show test data mean pm 3sigma as well
        data_y_true = [y_test, np.sqrt(1) * np.ones_like(y_test)]
        data_y_sample = [y_sample_mu, y_sample_sigma]
        label_y = ['true, $\mu\pm3\sigma$', 'sample, $\mu\pm3\sigma$']
    else:
        data_y_true = [y_test_noisy]
        data_y_sample = [y_sample_mu, y_sample_sigma]
        label_y = ['true', 'sample, $\mu\pm3\sigma$']
    if options['dataset'] == 'cascaded_tank':
        temp = 1024
    elif options['dataset'] == 'wiener_hammerstein':
        temp = 4000
    else:
        temp = 200
        
        # convert to numpy for evaluation

    '''
    # %% get the kalman loss
    A = np.array([[0.7, 0.8], [0, 0.1]])
    B = np.array([[-1], [0.1]])
    # C = np.array([[1], [0]]).transpose()
    C = np.array([[1], [-1]]).transpose()
    Q = np.sqrt(0.25) * np.identity(2)
    R = np.sqrt(1) * np.identity(1)
    y_kalman = run_kalman_filter(A, B, C, Q, R, u_test, y_test_noisy)
    # plt.plot(x, y_kalman.squeeze(), label='y_1(k) Kalman filter', linestyle='dashed', color='k')
    
    dv.plot_time_sequence_uncertainty(data_y_true,
                                      data_y_sample,
                                      label_y,
                                      options,
                                      batch_show=0,
                                      x_limit_show=[0, temp],
                                      path_general=path_general,
                                      file_name_general=file_name_general)
    
    # %% compute performance values for kalman filtering
    print('\nPerformance parameter of KF:')
    # compute VAF
    vaf_KF = de.compute_vaf(y_test_noisy, np.expand_dims(y_kalman, 0), doprint=True)
    # compute RMSE
    rmse_KF = de.compute_rmse(y_test_noisy, np.expand_dims(y_kalman, 0), doprint=True)

    '''
   
    # %% compute performance values

    # compute marginal likelihood (same as for predictive distribution loss in training)
    marginal_likeli = de.compute_marginalLikelihood(y_test_noisy, y_sample_mu, y_sample_sigma, doprint=True)
    # logLikelihood = de.compute_marginalLikelihood(y_test_noisy, y_sample_mu, y_sample_sigma, doprint=True)

    print('Performance parameter of NN model:')
    # compute VAF
    vaf = de.compute_vaf(y_test_noisy, y_sample_mu, doprint=True)

    # compute RMSE
    rmse = de.compute_rmse(y_test_noisy, y_sample_mu, doprint=True)


    print('\nModel: mean VAF = {}'.format(vaf))
    print('Model: mean RMSE = {}'.format(rmse))
    print('Model: mean log Likelihood = {}'.format(marginal_likeli))

    # print('\nKF: mean VAF = {}'.format(vaf_KF))
    # print('KF: mean RMSE = {}'.format(rmse_KF))
    
    print("y_test.shape,y_test_noisy.shape,y_sample_mu.shape,z_df.shape\n",y_test.shape,y_test_noisy.shape,y_sample_mu.shape,z.shape)
    
    
    # %% save the y and y^, y_mu and y_sigma
    output_save_path = path_general+file_name_general+".csv"
    data_output = pd.DataFrame()
    # ,'y_true':y_test.reshape(5000,),'y_true_with_noise': y_test_noisy.reshape(5000,), 'y_predict_mu': y_sample_mu.reshape(5000,),'y_predict_sigma':y_sample_sigma.reshape(5000,)
    
    value_list = [u_test, y_test, y_test_noisy, y_sample_mu, y_sample_sigma, z]
    value_name_list = ['u_test',"y_test", "y_test_noisy", "y_sample_mu", "y_sample_sigma", "z"]
    
    
    for v, v_name in zip(value_list,value_name_list):
        v=np.transpose(np.squeeze(v))
        if len(v.shape) == 1:
            column_name = [v_name+'_0']
        else:
            column_name = [v_name+'_'+str(i) for i in range(v.shape[-1])]

        v_df = pd.DataFrame(v, columns=column_name)
        for column_name in v_df.columns:
            data_output[column_name] = v_df[column_name]
    
    
    # data_output = {'y_true':y_test.reshape(5000,),'y_true_with_noise': y_test_noisy.reshape(5000,), 'y_predict_mu': y_sample_mu.reshape(5000,),'y_predict_sigma':y_sample_sigma.reshape(5000,),'z_predict_mu': z_sample_mu.reshape(5000,),'z_predict_sigma':z_sample_sigma.reshape(5000,)}
    print(y_test.shape,y_test_noisy.shape,y_sample_mu.shape,z.shape)
    df_output = pd.DataFrame(data_output)

    df_output.to_csv(output_save_path)

    # %% Collect data

    # options_dict
    options_dict = {'h_dim': options['model_options'].h_dim,
                    'z_dim': options['model_options'].z_dim,
                    'n_layers': options['model_options'].n_layers,
                    'seq_len_train': options['dataset_options'].seq_len_train,
                    'batch_size': options['train_options'].batch_size,
                    'lr_scheduler_nepochs': options['train_options'].lr_scheduler_nepochs,
                    'lr_scheduler_factor': options['train_options'].lr_scheduler_factor,
                    'model_param': num_model_param, }
    # test_dict
    test_dict = {'marginal_likeli': marginal_likeli,
                 'vaf': vaf,
                 'rmse': rmse}
    # dataframe
    df.update(options_dict)
    df.update(test_dict)

    return df
