import torch
import torch.optim as optim
import time
import math
import random
import os
from Utils.Baseline_model import baselineModel
from Utils.get_Dataloader import get_Dataloader


UE_list = [3, 4, 5, 7, 8]
for K in UE_list:
    print("---------- Train case with K = ", K, " ----------")

    # Define system configuration
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
    L = int(1 / 4 * M)  # Length of the pilot sequence
    B = 10  # Number of quantization bits
    SNR_dB = 10

    # Initialize the best rate
    best_rate_overall = 0
    best_model_state = None

    for run in range(5):
        print(f"Run {run + 1}/5")

        # Set different random seed for each run
        seed = run
        torch.manual_seed(seed)
        random.seed(seed)

        # Define the device
        device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

        # model
        model = baselineModel(K=K, M=M, UE_ant=UE_antenna, B=B, L=L, SNR=SNR_dB).to(device)

        # Define the optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=False)
        lambda1 = lambda epoch: 0.9995 ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

        # Training parameters
        n_epochs = 500  # Number of epochs
        alpha_init = 0.5
        increase = 1.001

        # Training
        tStart = time.time()
        train_rate_history = []
        val_rate_history = []
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(n_epochs):
                print('Epoch-{0} lr: {1}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
                alpha = max([alpha_init * (increase ** (epoch + 1)), 10])
                for x_batch in train_loader:
                    x_batch = x_batch[0].to(device)
                    model.train()
                    W, rate_list = model(x_batch, alpha)

                    sum_rate = rate_list[-1]
                    total_loss = -1 * sum_rate
                    train_rate_history.append(sum_rate.item())

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                # Validation
                with torch.no_grad():
                    model.eval()
                    val_rate_all = [0 for _ in range(K + 1)]
                    for x_val in val_loader:
                        x_val = x_val[0].to(device)
                        val_W, val_rate_list = model(x_val, alpha)

                        # Append the rate of each user
                        val_rate_list = [r.item() for r in val_rate_list]
                        val_rate_all = [x + y for x, y in zip(val_rate_all, val_rate_list)]

                    # Compute the sum rate
                    val_rate = [r / math.ceil(tot_val_sample / val_batch_size) for r in val_rate_all]
                    print("Rate: ", [format(r, '.4f') for r in val_rate])

                    # Save the best model
                    val_sum_rate = val_rate[-1]
                    val_rate_history.append(val_sum_rate)
                    if val_sum_rate > best_rate_overall:
                        best_rate_overall = val_sum_rate
                        best_model_state = model.state_dict()

                scheduler.step()

        # Save the best model across all runs
        if best_model_state is not None:
            dirs = 'Saved_model_temp/'
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            best_model_file = dirs + 'BaselineModel_Train_with_' + 'L=' + str(L) + '_B=' + str(B) + \
                              '_SNR=' + str(SNR_dB) + '_K=' + str(K) + '.pt'
            torch.save(best_model_state, best_model_file)
            print(f"Best model saved with rate: {best_rate_overall}")

        tEnd = time.time()
        training_time = tEnd - tStart
        print("It cost %f sec for training." % training_time)