%this m-file uses the acceptance/rejection 
%method to sample from the triangular density. 
clear;
clc;
rand('seed',sum(100*clock));
nkeep = 25000;

%--------------------------------
%triangular density
%--------------------------------
a = -1;
c = 0;
b = 1;

M = b-a;

triang_keep = zeros(nkeep,1);
counter =0;
    while counter < nkeep
        U = rand(2,1);
        U1 = U(1); % the proposal density q(x): Uniform[a, b]
        x = a + (b-a)*U1;   %Uniform[0,1] --> Uniform[a, b]
        
        U2 = U(2); % the uniform density U
        
        %the triangle density at x
        if x<=c
           pi_x = 2*(x-a)/(b-a)/(c-a);
        else
            pi_x = 2*(b-x)/(b-a)/(b-c);
        end
        
        if M/(b-a)*U2 < pi_x    % Mq(x) <pi(x), accepte
            counter = counter+1;  
            triang_keep(counter,1) = x;
        end;
    end;
    
%plotting the actual Triangular density
xgrid = linspace(-1,1,75);
size_x = size(xgrid);
density = xgrid;
for i =1 : size_x(1,2)
    x = xgrid(i);
    if x<=c
        density(i) = 2*(x-a)/(b-a)/(c-a);
    else
        density(i) = 2*(b-x)/(b-a)/(b-c);
    end
end

%plotting the Mq(x)
Mqx = M/(b-a);

%plot the results
[dom ran] = epanech2(triang_keep);
figure (1);
plot(dom,ran,'k');
xlabel('X');
ylabel('Triangular Density');
hold on;
plot(xgrid,density,'r.');

plot(xgrid,Mqx,'b.');
legend('estimated pdf using the samples', 'true pdf', 'the shape of Mq(x)');