clear;
clc;
rand('seed',sum(100*clock));

N_samples = 100;
n = 10;
m = 10;
X = 8;
Y = 6;

%generate samples for P1i, P2i
P1_N = betarnd( X+1, n-Y+1, N_samples, 1);
P2_N = betarnd( Y+1, m-Y+1, N_samples, 1);
delta = P1_N - P2_N;

delta_bar = mean(delta)

%hisogram pdf
figure(1)
hold on;
%nbins = sqrt(N_samples);
nbins = 11;;
[hist_f, xout] = hist(delta, nbins);
hist_f = hist_f/N_samples;
bandwidth = (max(delta)-min(delta))/nbins
hist_f = hist_f / bandwidth;
bar(xout, hist_f);

legend('histogram of delta');
     