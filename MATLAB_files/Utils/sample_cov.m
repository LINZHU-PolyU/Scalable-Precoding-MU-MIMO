function H_train_cov_cell = sample_cov(H_train)
% Generate sample covariance matrix using training dataset for LMMSE
% channel estimation.

[train_sample, K, ~] = size(H_train);
H_train_cov_cell = cell(K, 1);
for u = 1:K
    Hu = transpose(squeeze(H_train(:, u, :)));

    % Compute mean
    Hu_mean = sum(Hu, 2) / train_sample;

    % De-average
    Hu = Hu - repmat(Hu_mean, 1, train_sample);

    % Compute covariance matrix
    Hu_cov = 1/(train_sample - 1) * Hu * Hu';
    H_train_cov_cell{u} = Hu_cov;
end
end

