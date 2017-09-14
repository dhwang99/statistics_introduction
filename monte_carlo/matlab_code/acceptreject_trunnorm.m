%this m-file uses the acceptance/rejection 
%method to sample from the %truncnated normal density. 

clear all;
close all;

nkeep = 2500;
%--------------------------------
%Truncated Normal TN_[0,4] (1,1) density
%--------------------------------
mu = 1;
sigma = 1;

a = 0;
b = 4;

normcons = normcdf( (4-1)/1 ) - normcdf( (0-1)/1 ) ;
M = normpdf(1,1,1)*(b-a)/normcons;

tn_keep = zeros(nkeep,1);
counter = 0;
    while counter < nkeep
        U = rand(2,1);
        U1 = U(1);  % the proposal density q(x): Uniform[a, b]
        x = a + (b-a)*U1;   %Uniform[0,1] --> Uniform[a, b]
        
        U2 = U(2);
        
        % the triangle density at x
        pi_x = normpdf(x,mu,sigma)/normcons;
             
        if M/(b-a)*U2 < pi_x    % Mq(x) <pi(x), accepte
            counter = counter+1;
            tn_keep(counter,1) = x;
        end;
    end;
    
%plotting the actual Triangular density
xgrid2 = linspace(0,4,75);
densitya = normpdf(xgrid2,mu,sigma);
density2 = densitya/normcons;

%plotting the Mq(x)
Mqx = M/(b-a);

figure(1);
[dom2 ran2] = epanech2(tn_keep);
plot(dom2,ran2,'k');
xlabel('X');
ylabel('Truncated Normal Density');
hold on;
plot(xgrid2,density2,'r.');

plot(xgrid2,Mqx,'b.');
legend('estimated pdf using the samples', 'true pdf', 'the shape of Mq(x)');