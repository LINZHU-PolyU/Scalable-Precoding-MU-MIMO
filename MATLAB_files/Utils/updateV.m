function V = updateV(H, U, W, alpha, mu, fac1)
% Update V with found mu
[~, M, no_ue] = size(H);
V = cell(1, no_ue);
for ue = 1:no_ue
    alpha_ue = alpha(ue);
    H_ue = squeeze(H(:, :, ue));
    U_ue = U{ue};
    W_ue = W{ue};
    V_ue = pinv(fac1 + mu * eye(M) ) * ...
           alpha_ue * H_ue' * U_ue * W_ue;
    V{ue} = V_ue;
end
end

