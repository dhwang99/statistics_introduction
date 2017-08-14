%This function computes nonparametric density estimates 
%using the epanechnikov kernel and Silverman's (1985) 
%rule-of-thumb bandwidth selector.
%
%The only input is the scalar variable (say x) whose 
%denisty you want to estimate. Note that in this code, 
%it assumes x is a COLUMN VECTOR.
%
%The output is the domain (a grid of x values where the
%denisty estimates are obtained), and the range (the 
%density ordinates corresponding to each grid value). 
%The default here is to choose the lower and upper 
%bounds of the grid to be the min and max 
%of $x$, respectively and the grid to be 300 evenly
%spaced points between the min and max. This can, 
%and perhaps should be modified depending on the 
%application and sample size.
%
%The syntax is simply:
%		[domain range] = epanech2(x);
%and, of course, you can plot the result by using:
%plot(domain, range).
%
%NOTE THAT YOU CAN EASILY MANIPULATE THIS CODE TO 
%ALLOW THE BANDWIDTH AND GRID TO BE ADDED AS 
%ADDITIONAL INPUT VARIABLES.

function [dom,ran] = epanech2(draws);
gridsze = 300;
lowbd = min(draws); 
upbd = max(draws);
stderr = std(draws);
interquart = prctile(draws,75) - prctile(draws,25);
A = min(stderr,(interquart/1.34));
h = .9*A*(length(draws)^(-1/5));

dom = linspace(lowbd,upbd,gridsze)';
ran = zeros(gridsze,1); 

for i = 1:gridsze;
	a = -.5*sign( abs( (draws - dom(i,1))/h ) -sqrt(5) ) + .5;
	draws2 = nonzeros( draws.*a );
	tempp = zeros(length(draws2),1);	
		for j = 1:length(draws2);	
		tempp(j,1) = (.75/sqrt(5))*(1 -.2*( (draws2(j,1) - dom(i,1))/h )^2);
		end;
	clear draws2;		
ran(i,1) = (1/( (length(draws)*h) ) )*sum(tempp);
end;

	 	
