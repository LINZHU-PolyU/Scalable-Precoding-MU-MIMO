import torch
import torch.optim as optim
import time
import math
import random
import os
from torch.optim.lr_scheduler import _LRScheduler
from Utils.Proposed_GAT import myModel
from Utils.get_Dataloader import get_Dataloader


class WarmUpCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, T_max, T_warmup, eta_min=0, last_epoch=-1):
        self.T_max = T_max
        self.T_warmup = T_warmup
        self.eta_min = eta_min
        super(WarmUpCosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.T_warmup:
            return [base_lr * self.last_epoch / self.T_warmup for base_lr in self.base_lrs]
        else:
            k = 1 + math.cos(math.pi * (self.last_epoch - self.T_warmup) / (self.T_max - self.T_warmup))
            return [self.eta_min + (base_lr - self.eta_min) * k / 2 for base_lr in self.base_lrs]


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
L = int(1 / 4 * M)  # Length of the pilot sequence
B = 10  # Number of quantization bits
vq_dim = B  # Dimension of the extracted feature
SNR_dB_list = [20, 15, 10, 5, 0, -5]
n_quantizer = 1  # Number of quantizer
vq_b = int(B / n_quantizer)  # Number of bits per quantizer

for SNR_dB in SNR_dB_list:
    print("***** SNR dB: ", SNR_dB, ' *****')
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

        # Initialize the model
        model = myModel(vq_dim=vq_dim, vq_b=vq_b, n_ue=K, BS_ant=M, UE_ant=UE_antenna, n_quantizer=n_quantizer,
                        time_samples=L, SNR=SNR_dB).to(device)

        # Training parameters
        n_epochs = 500  # Number of epochs
        total_batches = int(math.ceil(tot_train_sample / train_batch_size))  # Number of batches per epoch
        replacement_num_batches = 10 * total_batches  # Replace unused codebooks every 10 epochs

        # Define the optimizer and scheduler
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=False)
        scheduler = WarmUpCosineAnnealingLR(optimizer=optimizer, T_max=n_epochs, T_warmup=30, eta_min=5e-5)

        # Training
        tStart = time.time()
        train_rate_history = []
        val_rate_history = []
        with torch.autograd.set_detect_anomaly(True):
            for epoch in range(n_epochs):
                print('Epoch-{0} lr: {1}'.format(epoch + 1, optimizer.param_groups[0]['lr']))
                for x_batch in train_loader:
                    x_batch = x_batch[0].to(device)
                    model.train()
                    W, vq_loss, rate_list = model(x_batch, train_mode=True)

                    sum_rate = rate_list[-1]
                    total_loss = -sum_rate + 1 * vq_loss
                    train_rate_history.append(sum_rate.item())

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                # Replace unused codebooks
                cur_tot_batches = (epoch + 1) * total_batches
                if (cur_tot_batches % replacement_num_batches == 0) and (cur_tot_batches != n_epochs * total_batches):
                    model.vq.replace_unused_codebooks(replacement_num_batches)

                # Validation
                with torch.no_grad():
                    model.eval()
                    val_vq_loss_list = []
                    val_rate_all = [0 for _ in range(K + 1)]
                    for x_val in val_loader:
                        x_val = x_val[0].to(device)
                        _, val_vq_loss, val_rate_list = model(x_val, train_mode=False)

                        # Append the rate and VQ loss of each user
                        val_rate_list = [r.item() for r in val_rate_list]
                        val_rate_all = [x + y for x, y in zip(val_rate_all, val_rate_list)]
                        val_vq_loss_list.append(val_vq_loss.item())

                    # Compute the average VQ loss and rate
                    val_vq_loss = sum(val_vq_loss_list) / len(val_vq_loss_list)
                    val_rate = [r / len(val_vq_loss_list) for r in val_rate_all]
                    print("VQ loss: ", format(val_vq_loss, '.4f'), ' | ',
                          "Rate: ", [format(r, '.4f') for r in val_rate])

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
            best_model_file = dirs + 'GAT_Train_with_' + 'L=' + str(L) + '_B=' + str(B) + '_SNR=' + str(SNR_dB) + \
                              '_K=' + str(K) + '.pt'
            torch.save(best_model_state, best_model_file)
            print(f"Best model saved with rate: {best_rate_overall}")

        tEnd = time.time()
        training_time = tEnd - tStart
        print("It cost %f sec for training." % training_time)
