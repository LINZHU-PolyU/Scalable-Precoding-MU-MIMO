function [ru] = getRate(u,hu,W,n_power)
% Get beamformer
wu = W(:,u);
wub = W(:,[1:u-1 u+1:end]);

% Compute rate
pu = abs(hu*wu)^2;
intu = sum(abs(hu*wub).^2) + n_power;
sinru = pu / intu;
ru = log2(1+sinru);
end

