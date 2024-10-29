function codebook = generate_codebook(B, H_train)
% Generate quantization codebook for each user based on the training
% dataset using generalized Lloyd algorithm.

[~, K, M] = size(H_train);
codebook = cell(1, K);
for u = 1:K
    fprintf('Computing codebook for User %d...\n', u)
    Hu = squeeze(H_train(:, u, :));
    Hu_unique = unique(Hu, 'rows', 'stable');
    N_sample = size(Hu_unique, 1);
    codebook_u = myLloyd(Hu_unique,M,2^B,N_sample,1000,1e-3);
    codebook{u} = codebook_u;
    fprintf('Done!\n')
end
end

