%this m-file uses the importance sampling 
%method to compute p(Z>3) 

clear;
clc;
rand('seed',sum(100*clock));
nkeep = 10000;

%--------------------------------
%Truncated Normal TN_(3,infinite] (0,1) density
%--------------------------------
c = 3;

%the basic monte carlo using standard normal as proposal
tn_keep_basic = zeros(nkeep,1);
h_n_basic = zeros(nkeep,1);
counter = 0;
    while counter < nkeep
        counter = counter+1;
        Z = normrnd(0,1,1,1);
        if Z > c
            h_n_basic(counter,1) = 1;
        else
            h_n_basic(counter,1) = 0;
        end;
    end;
I_cap_basic = mean(h_n_basic)
var_basic = var(h_n_basic)

%importance sampling using N(4,1) as proposal
mu = 4;
sigma = 1;
tn_keep_importance = zeros(nkeep,1);
h_n_importance = zeros(nkeep,1);
counter = 0;
    while counter < nkeep
        counter = counter+1;
        X = normrnd(mu, sigma);
        qX = normpdf(X, mu, sigma);
        if X>c
            h_n_importance(counter,1) = normpdf(X)/qX;
        else
            h_n_importance(counter,1) = 0;
        end;
    end;
I_cap_importance = mean(h_n_importance)
var_importance = var(h_n_importance)
