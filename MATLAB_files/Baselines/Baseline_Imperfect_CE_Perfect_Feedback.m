function sumr = Baseline_Imperfect_CE_Perfect_Feedback(Hu, L, H_train_cov_cell, P, n_power)
% Baseline scheme for Imperfect CE & Perfect Feedback.
% Each user estimates its own channel using LMMSE algorithm.
% User channel vector is assumed to be perfectly fed back to the BS

[K, M] = size(Hu);

%% Generate estimated channel using LMMSE
Hu_MMSE = zeros(K, M);
X = 1/sqrt(M) * dftmtx(M);  % DFT pilot for channel estimation
X = sqrt(1) * X(:, 1:L);  % Select the first L columns
for u = 1:K
    Hu_test = transpose(Hu(u, :));  % size: M x 1
    noise = sqrt(n_power/2) * (randn(L, 1) + 1i * randn(L,1));
    Y = X' * Hu_test + noise;  % Receivex pilot signal

    % LMMSE CE
    Hu_cov = H_train_cov_cell{u};  % Sample covariance matrix
    Hu_mmse = Hu_cov * X * inv(X' * Hu_cov * X + n_power*eye(L)) * Y;
    Hu_MMSE(u, :) = Hu_mmse.';  % Estimated channel
end

%% Compute precoding matrix and rate based on estimated channel with ZF
W_ZF_mmse = getZF(Hu_MMSE, P);
[sumr, ~] = getSumRate(Hu,W_ZF_mmse,n_power);
end

