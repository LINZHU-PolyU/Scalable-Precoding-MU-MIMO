function sumr = Baseline_WMMSE_CSIT(Hu, P, n_power)
% Baseline scheme for WMMSE: CSIT
% BS is assumed to have full CSIT. The precoding matrix is generated by
% WMMSE algorithm.

[K, M] = size(Hu);

%% Reshape H for WMMSE algorithm
Hu_wmmse = zeros(1,M,K);
for i=1:K
    Hu_wmmse(: , :, i) = Hu(i, :);
end

%% Run WMMSE with 100 iterations
max_iter = 100;  % Max. iterations
print = 0;  % 0: Not print / 1: print
W_WMMSE = WMMSE(Hu_wmmse, P, n_power, max_iter, 0);  % 100-iteration

%% Compute rate
[sumr, ~] = getSumRate(Hu,W_WMMSE,n_power);
end

