clear;
clc;
rand('seed',sum(100*clock));

N_samples = 10000;

h_N = zeros(N_samples, 1);
X_N = rand(N_samples, 1);

for  i = 1: N_samples
    h_N (i, 1)= X_N(i, 1)*X_N(i, 1)*X_N(i, 1);
end

I_cap = mean(h_N )
var_I_cap = var(h_N )
std_I_cap = sqrt(var_I_cap/N_samples )
     