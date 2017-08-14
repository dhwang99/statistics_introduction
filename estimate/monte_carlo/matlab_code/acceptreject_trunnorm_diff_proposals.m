%this m-file uses the acceptance/rejection 
%method to sample from the %truncnated normal density
%using different proposals. 
clear all;
clc;
close all;

rand('seed',sum(100*clock));

nkeep = 250;
%--------------------------------
%Truncated Normal TN_[c, +inf] (0,1) density
% c>0
%--------------------------------
mu = 0;
sigma = 1;
c = 1;
normcons = 1-  normcdf(c);

%plotting the actual density
figure(1);
hold on;
xgrid = linspace(c,c+3*sigma,75);
densitya = normpdf(xgrid,mu,sigma);
density = densitya/normcons;
plot(xgrid,density,'r.');

%using standard normal density as proposal
M = 1/normcons;

tn_keep1 = zeros(nkeep,1);
counter = 0;
total_counter = 0;
    while counter < nkeep
        total_counter = total_counter + 1;
        
        X = normrnd(mu, sigma);  % the proposal density q(x): N(0,1)
        if X>c    % Mq(x) <pi(x), accepte, in this example. pi(x) = Mq(x)
            counter = counter+1;
            tn_keep1(counter,1) = X;
        end;
    end;
disp('Observed Fraction of draws accepted wth standard Normal Source Density');
counter/total_counter

%plotting the Mq(x)
Mqx1 = M * densitya; %the same as the target distribution


%using  expoential density exp(beta)  as proposal
%beta = (c + sqrt(c*c+4) )/2;
%M = exp( (beta*beta - 2*beta*c)/2 )/sqrt(2*pi)/beta/normcons;     
beta =1;
M = exp(beta*beta/2)/sqrt(2*pi)/beta/normcons;     

tn_keep2 = zeros(nkeep,1);
counter = 0;
total_counter = 0;
    while counter < nkeep
        total_counter = total_counter + 1;
        
        X = exprnd(beta);  % the proposal density exp(beta)
        U = rand(1,1);
        
        %plot(X, U,'mo','MarkerSize',6);
        
        qX = exppdf(X, beta);
        if( X<c)
            pi_X = 0;
        else
            pi_X = normpdf(X, mu, sigma)/normcons;
       end
         
        if M*qX*U < pi_X    % Mq(x) <pi(x), accepte, in this example. proposal and the target distribution is the same, q(x)/pi(x) = 1
            counter = counter+1;
            tn_keep2(counter,1) = X;
            %plot(X, U,'ko','MarkerSize',6);
        end;
    end;
    
%str_info = sprintf('Observed Fraction of draws accepted wth exp(%d) Source Density is %d',beta, counter/total_counter);
%disp(str_info );

%plotting the Mq(x)
Mqx2 = M * exppdf(xgrid,beta);

[dom2 ran2] = epanech2(tn_keep2);
plot(dom2,ran2,'k');
xlabel('X');
ylabel('Truncated Normal Density');

plot(xgrid,Mqx2,'b.');
legend('true pdf','estimated pdf using the samples',  'the shape of Mq(x) of Exp(beta)');