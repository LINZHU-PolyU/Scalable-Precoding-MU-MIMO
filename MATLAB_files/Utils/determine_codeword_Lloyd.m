function [H_quant] = determine_codeword_Lloyd(C, Hu)
% Hu: M x K
H_quant = zeros(size(Hu));  % size: K x M
[N, ~] = size(C);  % N = 2^B/2
for u = 1:size(Hu,1)
    hu = Hu(u, :);  % size: 1 x M

    res = vecnorm(C - repmat(hu,N,1),2,2);

    [~, idx] = min(res);

    hu_quant = C(idx,:);

    H_quant(u,:) = hu_quant;
end
end

