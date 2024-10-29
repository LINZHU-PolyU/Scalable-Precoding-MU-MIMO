function [W_ZF] = getZF(H, P)
HH_h_inv = pinv(H * H');
W_ZF = H' * HH_h_inv;
[no_ue, M] = size(H);

% Normalize - Equal power allocation
v_norm = vecnorm(W_ZF, 2, 1);  % 1 x K
norm_fac = repmat(v_norm, M, 1);  % M x K
W_ZF = sqrt(P / no_ue) * W_ZF ./ norm_fac;
end

