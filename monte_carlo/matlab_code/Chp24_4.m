clear;
clc;
rand('seed',sum(100*clock));

N_samples = 100;
num_doses = 10;

doses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];                    % dose
num_animals = [15, 15, 15, 15, 15, 15, 15, 15, 15, 15];		% nunber of animals ni
Y = [0, 0, 2, 2, 8, 10, 12, 14, 15, 14];                    % number of dead Yi

delta_N = zeros(N_samples, 1);
counter = 0;
while counter < N_samples
    tcandidate_P = betarnd(Y+1, num_animals-Y+1);
    
    if tcandidate_P(1)<tcandidate_P(2) &  tcandidate_P(2)<tcandidate_P(3) & tcandidate_P(3)<tcandidate_P(4) & tcandidate_P(4)<tcandidate_P(5) & tcandidate_P(5)<tcandidate_P(6)...
       &tcandidate_P(6)<tcandidate_P(7) &  tcandidate_P(7)<tcandidate_P(8) & tcandidate_P(8)<tcandidate_P(9) & tcandidate_P(9)<tcandidate_P(10) 
       counter = counter + 1;
       for i=1:num_doses
           if tcandidate_P(i)> 0.5 
               break;
           end
       end
      
      delta_N(counter, 1) = doses(i);
    end
end
       
delta_bar = mean(delta_N);

%hisogram pdf
hist = zeros(num_doses, 1);
for i=1 : N_samples
   hist(delta_N(i, 1),1) = hist(delta_N(i, 1), 1) +1;
end

figure(1)
hold on;
hist = hist/N_samples;
bar(doses,hist);

legend('estimated density of detla');
     