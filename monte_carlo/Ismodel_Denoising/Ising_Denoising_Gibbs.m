% Denoising of image  using Gibbs sampling
% with an Ising Prior and a Gaussian likelihood.

seed = 4;
randn('state',seed)
rand ('state',seed)

sigma = 1; % noise level

% input matrix consisting of letter A. The body of letter
% A is made of 1's while the background is made of -1's.
F = imread('lettera.bmp'); %or some other path...
[M,N] = size(F);
d = double(F); d= 2.*((d-mean(mean(d)))>0)-1; %d either -1 1
% The body of letter
% A is made of 1's while the background is made of -1's.
y = d + sigma*randn(size(d)); %y: noisy letter A, size of the noise is sigma!

figure(1); clf
imagesc(y);colormap gray; axis square; 

%observation model
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
    
J = 5;
avgX = zeros(M,N);
X = Xinit;
maxIter = 200000;
for iter =1:maxIter
  % select a pixel at random
  ix = ceil( N * rand(1) ); iy = ceil( M * rand(1) );
  pos = iy + M*(ix-1);
  neighborhood = pos + [-1,1,-M,M];  
  neighborhood(find([iy==1,iy==M,ix==1,ix==N])) = [];
  
  % compute local conditional
  wi = sum( X(neighborhood) );
  if any(isnan(localEvidence(pos,:)))
    error(sprintf('no evidence at %d, %d', ix, iy))
  end
  
  p1  = exp(J*wi) * localEvidence(pos,onState);
  p0  = exp(-J*wi) * localEvidence(pos,offState);
  prob = p1/(p0+p1+eps);
  
  if rand < prob
    X(pos) = +1;
  else
    X(pos) = -1;
  end
  avgX = avgX+X;
  
  % plotting
  if rem(iter,1000) == 0,
    figure(fig);
    imagesc(X);  axis('square'); colormap gray; axis off;
    title(sprintf('sample %d', iter));
    drawnow
  end
end

