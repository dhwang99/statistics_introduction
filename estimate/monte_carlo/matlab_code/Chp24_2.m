clear;
clc;
rand('seed',sum(100*clock));

N_samples = 100000;
x = 2;

h_N = zeros(N_samples, 1);
X_N = normrnd(0, 1, N_samples, 1);

for  i = 1: N_samples
    if( X_N(i, 1) <= x )
       h_N(i, 1) =  1;  
    else
       h_N(i, 1) =  0;  	    
    end
end

I_cap = mean(h_N)
     