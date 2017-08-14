%Denoising of letter A: Ising Prior
 clear all
 close all
 
%-----------------figure defaults
disp('Denoising of A: Ising Prior')
randn('state',3) %set the seeds (state) to have 
rand ('state',3) %the constancy of results

% input matrix consisting of letter A. The body of letter
% A is made of 1's while the background is made of -1's.
F = imread('lettera.bmp'); %or some other path...
[M,N] = size(F);
sigma = 1;
d = double(F); d= 2.*((d-mean(mean(d)))>0)-1; %d either -1 1
% The body of letter
% A is made of 1's while the background is made of -1's.

%-----------------------------------------------
%observation model
y = d + sigma*randn(size(d)); %y: noisy letter A, size of the noise is sigma!
offState = 1; onState = 2;
mus = zeros(1,2); 
mus(offState) = -1; mus(onState) = +1;
sigmas = [sigma sigma];
[M N] = size(y);
Npixels = M*N;

localEvidence = ones(Npixels, 2); % 
for k=1:2
  localEvidence(:,k) = normpdf(y(:), mus(k), sigmas(k));
end

%double --> +1   -1 
[junk, guess] = max(localEvidence, [], 2);  
X = ones(M, N);
X(find(guess==offState)) = -1;
X(find(guess==onState)) = +1;
Xinit = X;

fig = figure(2); clf
figure(fig);
imagesc(X);  axis('square'); colormap gray; axis off;
title(sprintf('sample %d', 0));
drawnow

J = 5; %Reciprocal Temperature...
theta = Xinit;
iter=0;

adj = [-1 1 0 0; 0 0 -1 1];
maxIter = 200000;
for iter =1:maxIter
    ix = ceil( N * rand(1) ); iy = ceil( M * rand(1) );     % select one pixel
    pos = iy + M*(ix-1);                                    % j = (x-1)*M + y, 2D-->1D index
    thetap = -theta(pos);                                   % change the state of the selected pixel
    
    LikRat = exp( y(pos) * (thetap - theta(pos)) / sigma.^2);
    
    neighborhood = pos + [-1,1,-M,M];  
    neighborhood(find([iy==1,iy==M,ix==1,ix==N])) = [];
    
    disagree = sum(theta(neighborhood)~=theta(pos)); 
    disagreep = sum(theta(neighborhood)~=thetap);
    
    DelLogPr = 2 * J * (disagree - disagreep);
    
    alpha = exp(DelLogPr) * LikRat;
    if rand < alpha         % accepted
      theta(pos) = thetap;
    end
    
    iter = iter + 1;
     % plotting
  if rem(iter,1000) == 0
    figure(fig);
    imagesc(theta);  axis('square'); colormap gray; axis off;
    title(sprintf('sample %d', iter));
    drawnow
  end
end