function mu = search_mu(H, U, W, fac1, P)
% Keep searching mu
[~, ~, no_ue] = size(H);
[D,A] = eig(fac1);
Phi_tmp = 0;
for ue = 1:no_ue
    H_ue = squeeze(H(:, :, ue));
    U_ue = U{ue};
    W_ue = W{ue};
    Phi_tmp = Phi_tmp + H_ue' * U_ue * W_ue * W_ue * U_ue' * H_ue;
end
Phi = D' * Phi_tmp * D;

% Extract diagonal elements
Phi_diag = real(diag(Phi));
Phi_diag(Phi_diag<1e-6) = 0;  % Discard the extremly small values

A_diag = real(diag(A));
A_diag(A_diag<1e-6) = 0;      % Discard the extremly small values

% Compute power
mu_up = 1000;  % Upper bound for mu
mu_low = 0;   % Lower bound for mu

% Run bisection to find proper mu
while (mu_up - mu_low) >= 1e-6
    mu = (mu_up + mu_low)/2;
    P_tmp = sum(Phi_diag ./ (A_diag + mu).^2);

    % Update mu
    if P_tmp > P
        mu_low = mu;
    elseif P_tmp < P
        mu_up = mu;
    end
end
end

