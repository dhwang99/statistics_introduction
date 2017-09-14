%Realization of Binary Markov Random Field  by Gibbs
 clear all
 close all
%-----------------figure defaults
disp('Ising by Gibbs')
lw = 2;  
set(0, 'DefaultAxesFontSize', 16);
fs = 14;
msize = 5;
randn('state',3) %set the seeds (state) to have 
rand ('state',3) %the constancy of results

%----------------

pixelX = 256;
pixelY = 256;
beta = 0.25;
el = 0;
load klaus256; F =  (2.* (ima256 > 0.4) - 1);

figure(1)
colormap(gray)
imagesc(F)

%F = ( 2 .* ( rand( pixelX, pixelY ) > 0.5 ) - 1 );%if Klaus is not available...
%while 1,
for jj=1:30
for k = 1 : 10000
% Select a pixel at random
ix = ceil( pixelX * rand(1) );  
iy = ceil( pixelY * rand(1) );
Fc = F( iy, ix ); 
pos = ( ix - 1 ) * pixelY + iy; % Index of pixel
    neighborhood = pos + [-1 1 -pixelY pixelY]; % Find indicies of neighbours
    neighborhood( find( [iy == 1    iy == pixelY    ix == 1   ix == pixelX] ) ) = []; 
% Remove those outside picture
potential = sum(  F(neighborhood) );  
   if rand(1) < exp( - beta * potential  )/(  exp( - beta * potential  )  + exp(  beta * potential  ))
    F( iy, ix ) = -1;
   else 
    F( iy, ix ) = 1;
   end
el = el + 1;
end
figure(2); imagesc(F); colormap(gray); title(['Iteration #  ' num2str(el)]);
drawnow
end
 figure(2); imagesc(F); colormap(gray); title(['Iteration #  ' num2str(el)]);



