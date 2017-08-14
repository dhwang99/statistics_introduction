%Realization of Binary Markov Random Field  by Metropolis
 clear all
 close all
%-----------------figure defaults
disp('Ising by Metropolis')
lw = 2;  
set(0, 'DefaultAxesFontSize', 16);
fs = 14;
msize = 5;
randn('state',3) %set the seeds (state) to have 
rand ('state',3) %the constancy of results

pixelX = 256;
pixelY = 256;
J = 0.85;
el = 0;

F = ( 2 .* ( rand( pixelX, pixelY ) > 0.5 ) - 1 );

figure(1)
colormap(gray)
imagesc(F)


for jj = 1:500
    %while 1,
    for k = 1 : 10000
        % Select a pixel at random
        ix = ceil( pixelX * rand(1) ); 
        iy = ceil( pixelY * rand(1) );
        Fc = F( iy, ix ); %the value at position (ix, iy) 
        pos = ( ix - 1 ) * pixelY + iy; %  univariate index of pixel
        neighborhood = pos + [-1 1 -pixelY pixelY]; % Find indicies of neighbours
        neighborhood( find( [iy == 1    iy == pixelY    ix == 1   ix == pixelX] ) ) = []; 
    
        % pesky boundaries...thrash them
        nagree =  sum( Fc == F(neighborhood) ); 
        ndisagree = sum( Fc ~= F(neighborhood) );
        change = nagree - ndisagree;
        
        if rand(1) < exp( -2 * J * change ) % if change<0, proposal is taken wp 1
            F( iy, ix ) = -Fc;                %accept proposal 
        end
        el = el + 1;
    end

    figure(2); imagesc(63*F); colormap(gray); title(['Iteration #  ' num2str(el)]);
    drawnow
end

figure(2); imagesc(63*F); colormap(gray); title(['Iteration #  ' num2str(el)]);
