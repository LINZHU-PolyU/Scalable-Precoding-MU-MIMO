function V = WMMSE(H, P, n_power, max_iter, print)
% System setup
[R, M, no_ue] = size(H);
d = R;

% Parameters
epsilon = 1e-4;  % Halting condition
sigma2 = n_power;  % Noise power
% sigma2 = 1;
alpha = ones(no_ue, 1);  % Weighting facor for each UE

% Reshape channel for computing rate
Hu = zeros(no_ue, M);
for ue = 1:no_ue
    hu = squeeze(H(:, :, ue));
    Hu(ue, :) = hu;
end

% Initialze V
V = cell(1,no_ue);
for ue = 1:no_ue
    v = randn(M, d) + 1i * randn(M, d);
    V{ue} = sqrt( P / ( no_ue * trace(v * v') ) ) * v;
end 
[rate, ~] = getSumRate(Hu, cell2mat(V), n_power);

if print == 1
    fprintf('---------WMMSE is Running-------------\n');
    fprintf('Iter. No         Rate         Residual \n');
    fprintf('--------------------------------------\n');
    fprintf('iter %3i:       %6.3f       %5.2e \n', ...
        0,rate,0);
end

% WMMSE iterate
iter = 0;
while iter < max_iter
    iter = iter + 1;
    % Update U
    U = updateU(H, V, sigma2);

    % Update W
    W = updateW(H, V, U, d);

    % Update V
    fac1 = get_fac1(H, U, W, alpha);

    if rank(fac1) == M  % If fac1 is invertible
        % Compute V
        mu = 0;
        V = updateV(H, U, W, alpha, mu, fac1);

        % Check power constraint
        V_mtx = cell2mat(V);
        P_tmp = trace(V_mtx * V_mtx');
        if P_tmp <= P
            continue
        end
    end

    % Bisection to find proper mu
    mu = search_mu(H, U, W, fac1, P);

    % Update V with found mu
    V = updateV(H, U, W, alpha, mu, fac1);
    
    % Check convergence
    [rate_new, ~] = getSumRate(Hu, cell2mat(V), n_power);
    res = abs(rate_new - rate);
    if res <= epsilon
        if print == 1
            fprintf('iter %3i:       %5.3f       %5.2e Converged!\n\n', ...
                iter, rate_new, res);
        end
        break
    else
        if print == 1
            fprintf('iter %3i:       %5.3f       %5.2e \n', ...
                iter, rate_new, res);
        end
        rate = rate_new;
    end
end
V = cell2mat(V);

end

