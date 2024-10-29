clc; clear;


K_list = [3, 4, 5, 6, 7, 8];
sumr_ue = [];
SNR = 10;
P = 1;  % Total power
n_power = 10^(-SNR / 10);
for K = K_list
    fprintf('************ Test with UE = %d ************\n', K)

    %% Load data
    folder = ['Raytracing_K=' num2str(K) '/'];
    H_train = importdata(['../Data/' folder 'DATA_H_mu_train.mat']);
    H_test = importdata(['../Data/' folder 'DATA_H_mu_test.mat']);

    [train_sample,~,M] = size(H_train);
    [test_sample,~,~] = size(H_test);
    L = 1/4 * M;  % Length of pilot sequence


    %% Compute sample covariance matrix
    H_train_cov_cell = sample_cov(H_train);


    %% Train quantizer using generalized Lloyd algorithm
    B = 10;  % Number of Quant. bits
    codebook = generate_codebook(B, H_train);


    %% Transmission
    num_scheme = 6;  % # of schemes to be compared
    maxtrial = test_sample;
    sumr_all = zeros(maxtrial, num_scheme);
    trial_list = randperm(test_sample, maxtrial);
    for t = 1:length(trial_list)
        trial = trial_list(t);
        fprintf(' Current trial = %d \n',t);

        % UE Selection
        Hu = squeeze(H_test(trial, :, :)); % size: K x M

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
        sumr_all(t, :) = [sumr_wmmse; sumr_zf; sumr_mmse; sumr_quant;...
                               sumr_existing_DL; sumr_proposed];
    end 
    sumr_avg = squeeze(sum(sumr_all, 1)) ./ repmat(maxtrial, 1, num_scheme);
    sumr_ue = [sumr_ue; sumr_avg];
end

plot(K_list,sumr_ue(:, 1),'d--','LineWidth',1.5,'MarkerSize',9)
grid on
hold on
plot(K_list,sumr_ue(:, 2),'d--','LineWidth',1.5,'MarkerSize',9)
plot(K_list,sumr_ue(:, 3),'d--','LineWidth',1.5,'MarkerSize',9)
plot(K_list,sumr_ue(:, 4),'d--','LineWidth',1.5,'MarkerSize',9)
plot(K_list,sumr_ue(:, 5),'d--','LineWidth',1.5,'MarkerSize',9)
plot(K_list,sumr_ue(:, 6),'ko-','LineWidth',1.5,'MarkerSize',9)
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
xlabel('Number of users', 'Interpreter','latex'); 
ylabel('Average Sum Rate (bps/Hz)', 'Interpreter','latex');
xlim([K_list(1), K_list(end)]);
xticks(K_list);
set(gca,'looseInset',[0 0 0.02 0]);
set(gca,'TickLabelInterpreter','latex');
