function [t,x] = gillespiePP(T,x0,nu,propfun,PP)

R = size(nu,2);
t = 0; % time
rcT = zeros(R,1); % reaction-clock time

PPnext = zeros(R,1);
for i = 1:R
try
    PPnext(i) = PP{i}(1); % Assumes reaction times are sorted ascending
catch ME
    PPnext(i) = Inf;
end
end
nrcounter = ones(R,1);

x(:,1) = x0;

j=0;

while t(end)<T
    j=j+1;
    pf = propfun(x(:,j));
    
    rcWait = PPnext - rcT; % reaction-clock wait
    rtWait = rcWait./pf; % real-time wait
    
    [tau_f,reaction] = min(rtWait); % select the next reaction to fire

    rcT = rcT + tau_f*pf; % increase the reaction clock time
    
    t(j+1) = t(j) + tau_f; % increase the real-time time
    x(:,j+1) = x(:,j) + nu(:,reaction); % fire the reaction
    
	nrcounter(reaction) = nrcounter(reaction)+1; % increase the count of the fired reaction
    try
        PPnext(reaction) = PP{reaction}(nrcounter(reaction)); % get the times of all the next reactions
    catch ME
        % If there is no additional reaction to fire in this Poisson
        % process, just make one up?
        PPnext(reaction) = PPnext(reaction) + exprnd(1);
        %PPnext(reaction) = Inf;        
    end

end
