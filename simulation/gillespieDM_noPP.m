%function [t,x,unitPP] = gillespieDM(T,x0,nu,propfun)
function [t,x] = gillespieDM(T,x0,nu,propfun)
% Input simulation time T, stoichiometric matrix nu and handle to
% propensity function propfun. 
% Output a time vector, and the simulated trajectory x.
% pp{i} are the event times of the unit rate Poisson process underlying the
% firing of reaction i in this simulation.

i=1;
x(:,i) = x0;
t(i) = 0;

R = size(nu,2); % Number of reactions
reactionclock = zeros(R,1);
unitPP = cell(R,1);

% Gillespie simulation
while t(end)<T
    r = rand(2,1);
    
    p = propfun(x(:,i));
    p0 = sum(p);
    pp = cumsum(p)./p0;
    
    if p0==0
        t(i+1) = T;
        x(:,i+1) = x(:,i);
        return
    end
    
    tau_f = log(1/r(1))/p0; % Time of next event
    j = 1 + sum(pp<r(2));% Identity of next event
    
    % Record Poisson process
    %reactionclock  = reactionclock + tau_f*p;
    %unitPP{j} = [unitPP{j}, reactionclock(j)];
    
    % Record trajectory
    t(i+1) = t(i)+tau_f;
    x(:,i+1) = x(:,i)+nu(:,j);

    i = i+1;
end
