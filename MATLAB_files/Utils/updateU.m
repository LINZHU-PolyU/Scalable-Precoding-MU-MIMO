function U = updateU(H, V, sigma2)
% Function to update U
[R, ~, no_ue] = size(H);  % UE number
U = cell(1, no_ue);
for ue = 1:no_ue
    H_ue = squeeze(H(:, :, ue));
    V_ue = V{ue};
    V_mtx = cell2mat(V);
    V_gram = V_mtx * V_mtx';
    J_ue = H_ue * V_gram * H_ue' + sigma2 * eye(R);
    U_ue = inv(J_ue) * H_ue * V_ue;
    U{ue} = U_ue;
end
end

