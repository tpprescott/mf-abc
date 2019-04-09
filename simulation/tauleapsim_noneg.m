function [t,x,unitPP] = tauleapsim_noneg(T,x0,nu,propfun,diffpropfun,tau_upper,nc,eps)
% Input simulation time T, stoichiometric matrix nu and handle to
% propensity function propfun, and the common tau leap parameter.
% Output a time vector, and the simulated trajectory x.
% unitPP.IntervalEventCounts are the number of reaction firings that happen
% in each interval of the underlying unit rate Poisson process, with
% interval lengths defined by unitPP.IntervalLengths.
% delay allows for an accurate transient before starting the tau leap
% simulation


R = size(nu,2); % Number of reactions
unitPP.exactSteps = cell(R,1);

%% Simulate delay to T using tau-leap
t = 0;
t_tauleap(:,1) = t;
x_tauleap(:,1) = x0;

j = 0;

intT = zeros(R,1);
intK = zeros(R,1);

while t<T
    
    j = j+1;
    tau = tau_upper;
    
    p = propfun(x_tauleap(:,j));
    p0 = sum(p);
    
    L = min(x_tauleap(:,j)./(abs(nu).*(nu<0)))';
    crit_r = (p>0).*(L<nc);
    
    dp = diffpropfun(x_tauleap(:,j));
    if sum(1-crit_r)>0
        F = dp*nu(:,crit_r==0);
        mu = F*p(crit_r==0);
        s2 = (F.^2)*p(crit_r==0);
        tau_slower = min(min(eps*p0./abs(mu), ((eps*p0)^2)./s2));
        tau = min(tau,tau_slower);
    end
    
    if sum(crit_r) == 0
        K_noncrit = poissrnd(p*tau);
        K = K_noncrit;
    else
        pc = sum(p.*crit_r);
        tau_crit = exprnd(1/pc);
        K_crit = zeros(size(p));
        
        if tau_crit<tau
            % Exact timestep
            tau = tau_crit;
            
            % Choose which (critical) reaction fired
            ppc = cumsum(p.*crit_r)/pc;
            reaction = 1+sum(ppc<rand(1));
            K_crit(reaction) = 1;
            
            % Record exact information to unitPP
            unitPP.exactSteps{reaction} = [unitPP.exactSteps{reaction}; sum(intT(reaction,:))+tau*p(reaction)];
        end
        % Allow non-critical reactions to fire during tau-leap
        K_noncrit = poissrnd(tau*p.*(1-crit_r));
        
        % Add the firings together
        K = K_noncrit + K_crit;
        
    end
    
    t = t + tau;
    
    % Record unitPP partial information
    intK(:,j) = K_noncrit;
    intT(:,j) = tau*p;
    
    % Record trajectory
    x_tauleap(:,j+1) = x_tauleap(:,j) + nu*K;
    t_tauleap(:,j+1) = t;
    %x_tauleap(:,i+1) = x_tauleap(:,i+1).*(x_tauleap(:,i+1) >= 0);
end

t = t_tauleap;
x = x_tauleap;

unitPP.IntervalLengths = intT;
unitPP.IntervalEventCounts = intK;

end
