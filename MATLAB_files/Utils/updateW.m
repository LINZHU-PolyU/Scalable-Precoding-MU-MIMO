function W = updateW(H, V, U, d)
% Function to update W
no_ue = size(H, 3);  % UE number
W = cell(1, no_ue);
for ue = 1:no_ue
    H_ue = squeeze(H(:, :, ue));
    V_ue = V{ue};
    U_ue = U{ue};
    W_ue = inv( eye(d) - U_ue' * H_ue * V_ue );
    W{ue} = W_ue;
end
end

