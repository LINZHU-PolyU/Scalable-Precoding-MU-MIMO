import torch
from torch.utils.data import TensorDataset, DataLoader
import scipy.io as sio


def get_Dataloader(input_dict):

    # Get parameters
    train_file = input_dict['train_file']
    val_file = input_dict['val_file']
    test_file = input_dict['test_file']
    train_batch_size = input_dict['train_batch_size']
    val_batch_size = input_dict['val_batch_size']
    test_batch_size = input_dict['test_batch_size']

    # Load data
    mat = sio.loadmat(train_file)
    H_train = mat['H_mu_train']  # Batch x K x M

    mat = sio.loadmat(val_file)
    H_val = mat['H_mu_val']  # Batch x K x M

    mat = sio.loadmat(test_file)
    H_test = mat['H_mu_test']  # Batch x K x M

    # Change datatype
    H_train = H_train.astype('complex64')
    H_val = H_val.astype('complex64')
    H_test = H_test.astype('complex64')

    # Get total number of samples
    tot_train_sample = len(H_train)
    tot_val_sample = len(H_val)
    tot_test_sample = len(H_test)

    # Transform to tensor
    H_train = torch.tensor(H_train, dtype=torch.complex64)
    H_val = torch.tensor(H_val, dtype=torch.complex64)
    H_test = torch.tensor(H_test, dtype=torch.complex64)

    # Create TensorDatasets
    train_dataset = TensorDataset(H_train)
    val_dataset = TensorDataset(H_val)
    test_dataset = TensorDataset(H_test)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    # Return
    output = {}
    output['tot_train_sample'] = tot_train_sample
    output['tot_val_sample'] = tot_val_sample
    output['tot_test_sample'] = tot_test_sample
    output['train_loader'] = train_loader
    output['val_loader'] = val_loader
    output['test_loader'] = test_loader

    return  output
