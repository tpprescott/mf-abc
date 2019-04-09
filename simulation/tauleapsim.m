function [t,x,unitPP] = tauleapsim(T,x0,nu,propfun,tau)
% Input simulation time T, stoichiometric matrix nu and handle to
% propensity function propfun, and the common tau leap parameter.
% Output a time vector, and the simulated trajectory x.
% unitPP.IntervalEventCounts are the number of reaction firings that happen
% in each interval of the underlying unit rate Poisson process, with
% interval lengths defined by unitPP.IntervalLengths.

R = size(nu,2); % Number of reactions
unitPP.exactSteps = cell(R,1); % Legacy of idiocy

t_tauleap = 0:tau:T+tau;
x_tauleap = zeros(length(x0),length(t_tauleap));
x_tauleap(:,1) = x0;

R = size(nu,2); % Number of reactions
intT = zeros(R,1);
intK = zeros(R,1);

for i=1:length(t_tauleap)-1
    p = propfun(x_tauleap(:,i));
    K = poissrnd(p*tau);
    
    % Record unitPP partial information
    intK(:,i) = K;
    intT(:,i) = tau*p;
    
    % Record trajectory
    x_tauleap(:,i+1) = x_tauleap(:,i) + nu*K;
    x_tauleap(:,i+1) = x_tauleap(:,i+1).*(x_tauleap(:,i+1) >= 0);
end

t = t_tauleap;
x = x_tauleap;

unitPP.IntervalLengths = intT;
unitPP.IntervalEventCounts = intK;

end
