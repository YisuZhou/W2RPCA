function [A_hat] = RPCA_iALM(D, NSig,Par)
%   Inputs:
%       D            --- batch image matrix
%		NSig		 --- 3p^2 x 1 dimensional vector of weights
%
%   Outputs: 
%       A_hat        --- low-rank component
%--------------------------------------------------------------------------
% objective function:
% min ||A||_* + lambda ||WE||_1 s.t. D = A + E
% Lagrangian function:
% L(A, E, Y) = ||A||_* + lambda ||WE||_1 + <Y, D - A - E> + mu/2 ||D - A - E||_F^2
%--------------------------------------------------------------------------
[m, n] = size(D);
lambda = 1 / sqrt(m);

%tolerance = 1e-7;%use it to test whether it is converged. Actually,it doesnot perform well.

% Intialize the weight matrix W
mNSig = min(NSig);
W = (mNSig+eps) ./ (NSig+eps);
W_R=mean(W(1:Par.ps2));
W_G=mean(W(Par.ps2+1:2*Par.ps2));
W_B=mean(W(2*Par.ps2+1:3*Par.ps2));

Y = D;

norm_two = norm(Y, 2);
norm_inf = norm(Y(:), inf) / lambda;
dual_norm = max(norm_two, norm_inf);
Y = Y / dual_norm;

A_hat = zeros(m, n);
E_hat = zeros(m, n);

maxIter = 4;%reach this value then end.
mu=1.001;
rho = 1.0001;

mu_max = 1e+10;
constant =  2 * sqrt(2);  
TempC  = constant * sqrt(n) * mNSig^2;

iter = 0;
converged = false;
%% start optimization
while ~converged       
    iter = iter + 1;
    
    % updataA,
	%min_{A} ||A||_*,w + 0.5 * mu * ||D-L-S + 1/mu * Y)||_F^2
     temp_T = D - E_hat + (1/mu)*Y;
     temp_T(isnan(temp_T)) = 0;
     temp_T(isinf(temp_T)) = 0;     
     A_hat =  prox_reweighted_WNNP(temp_T, 2/mu*TempC, eps);
    %=============================== 
    
    % update E,
	% min_x  lambda||WS||_1 + mu/2*||D-L-S+1/mu*Y||_F^2
	temp_T = D - A_hat + (1/mu)*Y;
	temp_TR=temp_T(1:Par.ps2,:);
	temp_TG=temp_T(Par.ps2+1:2*Par.ps2,:);
	temp_TB=temp_T(2*Par.ps2+1:3*Par.ps2,:);
	E_hat(1:Par.ps2,:)=rpca_soft(temp_TR,W_R*lambda/mu);
	E_hat(Par.ps2+1:2*Par.ps2,:)=rpca_soft(temp_TG,W_G*lambda/mu);
	E_hat(2*Par.ps2+1:3*Par.ps2,:)=rpca_soft(temp_TB,W_B*lambda/mu);
	%=============================== 
	
    % update Y,
    Z = D - A_hat - E_hat;
    Y = Y + mu * Z;
	%=============================== 
    
    % update mu,   
    mu = min(mu*rho, mu_max);    
	%=============================== 

	%reach the iteration number.
    if ~converged && iter >= maxIter
        converged = true ;       
    end  
end

