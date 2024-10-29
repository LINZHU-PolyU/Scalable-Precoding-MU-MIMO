function [C] = myLloyd(H,M,N,N_sample,maxit,epsilon)
% Initialization - 2
idx_list = floor(linspace(1,N_sample,N));
% idx_list = randperm(N_sample, N);
C = H(idx_list, :);
for iter = 1:maxit
    fprintf('----- Current iter = %d ----- \n',iter);
    % Assign groups
    W = cell(1,N);
    for n = 1:N_sample
        hn = H(n, :);
        tmp = C - hn;
        d = vecnorm(tmp, 2, 2); % Distance list
        [~,idx] = min(d);
        W{idx} = [W{idx}; hn];
    end
    
    % Compute new centroids
    C_new = zeros(N, M);
    for n = 1:N
        tmp = W{n};
        if isempty(tmp) == 0  % Not empty
            cur_sum = 0;
            for i = 1:size(tmp,1)
                row_i = tmp(i, :);
                cur_sum = cur_sum + row_i;
            end
            cent_new = cur_sum / size(tmp,1);
            C_new(n, :) = cent_new;
        else
            cent_new = H(randi(N_sample), :);
            C_new(n, :) = cent_new;
        end
    end
    
    % Check convergence
    res = norm(C - C_new, 'fro');
    if res <= epsilon
        break
    else
        C = C_new;
        fprintf('res = %.5f \n', res)
    end
end
end

