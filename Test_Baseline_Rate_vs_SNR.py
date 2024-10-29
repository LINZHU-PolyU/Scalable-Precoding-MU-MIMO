import torch
import scipy.io as sio
import time
import numpy as np
import os
from Utils.Baseline_model import baselineModel
from Utils.get_Dataloader import get_Dataloader


# Define system configuration
K = 6  # Number of users
M = 16  # Number of BS antennas
scenario = 'Raytracing' + '_K=' + str(K) + '/'

# Define data file
train_file = 'Data/' + scenario + 'DATA_H_mu_train.mat'
val_file = 'Data/' + scenario + 'DATA_H_mu_val.mat'
test_file = 'Data/' + scenario + 'DATA_H_mu_test.mat'

# Define batch size
train_batch_size = 500
val_batch_size = 1000
test_batch_size = 1000

# Define input_dict
input_dict = {}
input_dict['train_file'] = train_file
input_dict['val_file'] = val_file
input_dict['test_file'] = test_file
input_dict['train_batch_size'] = train_batch_size
input_dict['val_batch_size'] = val_batch_size
input_dict['test_batch_size'] = test_batch_size

# Get samples number and dataloader
output = get_Dataloader(input_dict)
train_loader = output['train_loader']
val_loader = output['val_loader']
test_loader = output['test_loader']
tot_train_sample = output['tot_train_sample']
tot_val_sample = output['tot_val_sample']
tot_test_sample = output['tot_test_sample']

# Network params
UE_antenna = 1  # MISO case
L = int(1/4 * M)  # Length of the pilot sequence
B = 10  # Number of quantization bits
SNR_dB_list = [20, 15, 10, 5, 0, -5]

for SNR_dB in SNR_dB_list:
    print("***** SNR dB: ", SNR_dB, ' *****')

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

    # Testing data
    with torch.no_grad():
        tStart = time.time()
        test_model = baselineModel(n_ue=K, BS_ant=M, UE_ant=UE_antenna, B=B, time_samples=L, SNR=SNR_dB).to(device)

        # Load the trained model
        dirs = 'Saved_model/'
        # dirs = 'Saved_model_temp/'
        test_model_file = dirs + 'BaselineModel_Train_with_' + 'L=' + str(L) + '_B=' + str(B) + \
                              '_SNR=' + str(SNR_dB) + '_K=' + str(K) + '.pt'
        test_model.load_state_dict(torch.load(test_model_file))
        test_model.eval()

        # Testing
        W_list = []
        sum_rate_list = []
        for x_test in test_loader:
            x_test_batch = x_test[0].to(device)

            # Perform inference
            W, rate_list = test_model(x_test_batch, alpha=10)

            # Move results to CPU to free GPU memory
            W = W.to('cpu').detach().numpy()  # B x M x K
            rate_list = [r.to('cpu').detach().numpy() for r in rate_list]
            sum_rate = rate_list[-1]

            # Append results to list
            W_list.append(W)
            sum_rate_list.append(sum_rate)

        # Reshape results
        W = np.concatenate(W_list, axis=0)  # B x M x K
        sum_rate = np.mean(sum_rate_list)  # scalar

        print("********** Testing data: **********\n",
              "SumRate: {:.4f}".format(sum_rate),
              "\n***********************************\n")

        tEnd = time.time()
        print("It cost %f sec." % ((tEnd - tStart) / tot_test_sample))

    # Save the results
    res_dirs = 'Res/' + scenario
    if not os.path.exists(res_dirs):
        os.makedirs(res_dirs)
    rate_file = res_dirs + 'SumRate_L=' + str(L) + '_B=' + str(B) + '_SNR=' + str(SNR_dB) + '_K=' + str(K) + \
                '_BaselineModel.mat'
    sio.savemat(rate_file, {'data': sum_rate})

    BF_file = res_dirs + 'W_L=' + str(L) + '_B=' + str(B) + '_SNR=' + str(SNR_dB) + '_K=' + str(K) + \
              '_BaselineModel.mat'
    sio.savemat(BF_file, {'data': W})
