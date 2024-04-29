from data.base import DataLoaderExt
# from data.cascaded_tank import create_cascadedtank_datasets
# from data.f16gvt import create_f16gvt_datasets
from data.narendra_li import create_narendra_li_datasets
from data.toy_lgssm import create_toy_lgssm_datasets
from data.wiener_hammerstein import create_wienerhammerstein_datasets
from data.toy_lgssm_5_pre import create_toy_lgssm_5_datasets
from data.toy_lgssm_2dy_5_pre import create_toy_lgssm_2dy_5_datasets
from data.f16gvt import create_f16gvt_datasets
from data.IndustRobo import create_industrobo_datasets


def load_dataset(dataset, dataset_options, train_batch_size, test_batch_size, **kwargs):
    """Not used datasets: F16 and Cascadedtank"""
    """if dataset == 'cascaded_tank':
        dataset_train, dataset_valid, dataset_test = create_cascadedtank_datasets(dataset_options.seq_len_train,
                                                                                  dataset_options.seq_len_val,
                                                                                  dataset_options.seq_len_test)
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
    """
    if dataset == 'narendra_li':
        dataset_train, dataset_valid, dataset_test = create_narendra_li_datasets(dataset_options.seq_len_train,
                                                                                 dataset_options.seq_len_val,
                                                                                 dataset_options.seq_len_test,
                                                                                 **kwargs)
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
    elif dataset == 'toy_lgssm':
        dataset_train, dataset_valid, dataset_test = create_toy_lgssm_datasets(dataset_options.seq_len_train,
                                                                               dataset_options.seq_len_val,
                                                                               dataset_options.seq_len_test,
                                                                               **kwargs)
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
    
    elif dataset == 'toy_lgssm_5_pre':
        dataset_train, dataset_valid, dataset_test = create_toy_lgssm_5_datasets(dataset_options.seq_len_train,
                                                                               dataset_options.seq_len_val,
                                                                               dataset_options.seq_len_test,
                                                                               **kwargs)
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
        
    elif dataset == 'toy_lgssm_2dy_5_pre':
        dataset_train, dataset_valid, dataset_test = create_toy_lgssm_2dy_5_datasets(dataset_options.seq_len_train,
                                                                               dataset_options.seq_len_val,
                                                                               dataset_options.seq_len_test,
                                                                               **kwargs)
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)

    elif dataset == 'wiener_hammerstein':
        dataset_train, dataset_valid, dataset_test = create_wienerhammerstein_datasets(dataset_options.seq_len_train,
                                                                                       dataset_options.seq_len_val,
                                                                                       dataset_options.seq_len_test,
                                                                                       **kwargs)
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
        
    elif dataset == 'f16gvt':
        dataset_train, dataset_valid, dataset_test = create_f16gvt_datasets(dataset_options.seq_len_train,
                                                                            dataset_options.seq_len_val,
                                                                            dataset_options.seq_len_test,
                                                                            dataset_options.input_type,
                                                                            dataset_options.input_lev,
                                                                            **kwargs
                                                                            )
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)


    elif dataset == 'industrobo':
        print("dataset_options.seq_stride," ,dataset_options.seq_stride)
        dataset_train, dataset_valid, dataset_test = create_industrobo_datasets(dataset_options.seq_len_train,
                                                                            dataset_options.seq_len_val,
                                                                            dataset_options.seq_len_test,
                                                                            dataset_options.seq_stride,
                                                                            dataset_options.y_dim,
                                                                            dataset_options.input_channel,
                                                                            **kwargs
                                                                            )
        # Dataloader
        loader_train = DataLoaderExt(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=1)
        loader_valid = DataLoaderExt(dataset_valid, batch_size=test_batch_size, shuffle=False, num_workers=1)
        loader_test = DataLoaderExt(dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=1)
    else:
        raise Exception("Dataset not implemented: {}".format(dataset))

    return {"train": loader_train, "valid": loader_valid, "test": loader_test}
