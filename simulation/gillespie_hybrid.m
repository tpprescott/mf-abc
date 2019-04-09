function [t,x,unitPP] = gillespie_hybrid(T,x0,nu,propfun,xcorrfun)
% Input simulation time T, stoichiometric matrix nu and handle to
% propensity function propfun.
% Output a time vector, and the simulated trajectory x.
% pp{i} are the event times of the unit rate Poisson process underlying the
% firing of reaction i in this simulation.
% ADAPTED FROM gillespieDM TO TAKE INTO ACCOUNT EVOLUTION OF STATE WITH
% DETERMINISTIC PROCESSES

i=1;
x(:,i) = x0;
t(i) = 0;

R = size(nu,2); % Number of reactions
reactionclock = zeros(R,1);
unitPP = cell(R,1);

% Gillespie simulation
while t(end)<T
    r1 = rand(1,1);
    
    p = propfun(x(:,i));
    p0 = sum(p);
    
    tau_s = log(1/r1)/p0; % Time to wait to next slow event
    
    % Correct based on deterministic process: next fast event and updated
    % time
    [tau_f, xcorrval] = xcorrfun(x(:,i));
    
    % Record time
    tau = min(tau_f,tau_s);
    t(i+1) = t(i)+tau;
    
    % Truncate at T --- nothing happened.
    if t(end)>T
        t(end) = T;
        x(:,i+1) = x(:,i);
        return
    end
    
    reactionclock  = reactionclock + tau*p;
    
    if tau_s < tau_f
        
        r2 = rand(1,1);
        pp = cumsum(p)./p0;
        j = 1 + sum(pp<r2);% Identity of next slow event
        
        x(:,i+1) = x(:,i) + nu(:,j);
        
        % Record Poisson process
        unitPP{j} = [unitPP{j}, reactionclock(j)];
    else
        x(:,i+1) = x(:,i) + xcorrval;
    end
    
    i = i+1;
end
