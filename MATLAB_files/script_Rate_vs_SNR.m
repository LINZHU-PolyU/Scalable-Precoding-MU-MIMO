clc; clear;


%% Load data
K = 6;  % Number of users
folder = ['Raytracing_K=' num2str(K) '/'];
H_train = importdata(['../Data/' folder 'DATA_H_mu_train.mat']);
H_test = importdata(['../Data/' folder 'DATA_H_mu_test.mat']);

SNR_list = -5:5:20;
[train_sample,~,M] = size(H_train);
[test_sample,~,~] = size(H_test);
L = 1/4 * M;  % Length of pilot sequence


%% Compute sample covariance matrix
H_train_cov_cell = sample_cov(H_train);  


%% Train quantizer using Lloyd algorithm
B = 10;  % Number of Quant. bits
codebook = generate_codebook(B, H_train);  


%% Transmission
num_scheme = 6;  % # of schemes to be compared
maxtrial = test_sample;
sumr_all = zeros(maxtrial, num_scheme, length(SNR_list));
trial_list = 1:maxtrial;
for t = 1:length(trial_list)
    trial = trial_list(t);
    fprintf(' Current trial = %d \n',t);

    % UE Selection
    Hu = squeeze(H_test(trial, :, :)); % size: K x M

    % Compute rate
    sumr4SNR_list = zeros(num_scheme, length(SNR_list));
    r4SNR_list = zeros(num_scheme, 2, length(SNR_list));
    for s = 1:length(SNR_list)
        SNR = SNR_list(s);
        P = 1;  % Total power
        n_power = 10^(-SNR / 10);
        
        % Baseline methods
        sumr_wmmse = Baseline_WMMSE_CSIT(Hu, P, n_power);  % WMMSE: CSIT

        sumr_zf = Baseline_ZF_CSIT(Hu, P, n_power); % ZF: CSIT

        sumr_mmse = Baseline_Imperfect_CE_Perfect_Feedback(Hu, L, ...
            H_train_cov_cell, P, n_power);  % ZF: Imperfect CE & Perfect Feedback

        sumr_quant = Baseline_Perfect_CE_Imperfect_Feedback(Hu, ...
            codebook, P, n_power); % ZF: Perfect CE & Imperfect Feedback

        sumr_existing_DL = importdata(['../Res/' folder ...
                              'SumRate_L=' num2str(L) '_' ...
                              'B=' num2str(B) '_' ...
                              'SNR=' num2str(SNR) '_'...
                              'K=' num2str(K) '_BaselineModel.mat']);
        
        % Proposed method
        sumr_proposed = importdata(['../Res/' folder ...
                              'SumRate_L=' num2str(L) '_' ...
                              'B=' num2str(B) '_' ...
                              'SNR=' num2str(SNR) '_'...
                              'K=' num2str(K) '_GAT.mat']);
        
        % Store all the sum rate
        sumr4SNR_list(:, s) = [sumr_wmmse; sumr_zf; sumr_mmse; sumr_quant;...
                               sumr_existing_DL; sumr_proposed];
    end
    sumr_all(t, :, :) = sumr4SNR_list;
end
sumr_avg = squeeze(sum(sumr_all, 1)) ./ repmat(maxtrial, num_scheme, length(SNR_list));


%% Draw figure
plot(SNR_list,sumr_avg(1, :),'d--','LineWidth',1.5,'MarkerSize',9)
grid on
hold on
plot(SNR_list,sumr_avg(2, :),'d--','LineWidth',1.5,'MarkerSize',9)
plot(SNR_list,sumr_avg(3, :),'d--','LineWidth',1.5,'MarkerSize',9)
plot(SNR_list,sumr_avg(4, :),'d--','LineWidth',1.5,'MarkerSize',9)
plot(SNR_list,sumr_avg(5, :),'d--','LineWidth',1.5,'MarkerSize',9)
plot(SNR_list,sumr_avg(6, :),'ko-','LineWidth',1.5,'MarkerSize',9)
hold off
legend(...
       'WMMSE: CSIT', ...
       'ZF: CSIT', ...
       'ZF: Imperfect CE \& Perfect Feedback', ...
       'ZF: Perfect CE \& Imperfect Feedback', ...
       'DL Method in [3]', ...
       'Proposed Method', ...
       ...
       'Location', 'northwest', ...
       'Interpreter', 'latex')
xlabel('SNR (dB)', 'Interpreter','latex'); 
ylabel('Sum Rate (bps/s/Hz)', 'Interpreter','latex');
xlim([SNR_list(1), SNR_list(end)]);
xticks(SNR_list);
set(gca,'looseInset',[0 0 0.02 0]);
set(gca,'TickLabelInterpreter','latex');

