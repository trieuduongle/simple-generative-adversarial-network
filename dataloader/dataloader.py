from .dataloader_moving_mnist import load_data as load_mmnist

def load_data(dataname,batch_size, val_batch_size, data_root, num_workers, **kwargs):
    if dataname == 'mmnist':
        return load_mmnist(batch_size, val_batch_size, data_root, num_workers)
    else:
        raise Exception(f'data loader {dataname} is not supported')