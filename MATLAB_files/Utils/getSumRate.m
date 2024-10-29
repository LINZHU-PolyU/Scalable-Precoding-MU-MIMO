function [sumr, r] = getSumRate(Hu,W,n_power)
% Compute sum rate
no_ue = size(Hu, 1);
sumr = 0;
r = [];
for u = 1:no_ue
    hu = Hu(u,:);
    ru = getRate(u,hu,W,n_power);
    sumr = sumr + ru;
    r(u) = ru;
end
end

