function fac1 = get_fac1(H, U, W, alpha)
[~, ~, no_ue] = size(H);
fac1 = 0;
for l = 1:no_ue
    alpha_l = alpha(l);
    H_l = squeeze(H(:, :, l));
    U_l = U{l};
    W_l = W{l};
    fac1 = fac1 + alpha_l * H_l' * U_l * W_l * U_l' * H_l;
end
end

