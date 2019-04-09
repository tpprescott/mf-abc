function [t,x] = gillespiePP_partition(T,x0,propfun,nu,slowPP,slowidx)

R = size(nu,2);
Rs = numel(slowPP);
[slowflag,PPloc] = ismember(1:R,slowidx);

t = 0; % real time
rcT = zeros(R,1); % reaction clock time
nrcounter = zeros(R,1);

PPnext = zeros(R,1);

for r = 1:R
    if slowflag(r)
        try
            PPnext(r) = slowPP{PPloc(r)}(1); % Assumes reaction times are sorted ascending
            nrcounter(r) = 1;
        catch ME
            PPnext(r) = Inf; % If there are no firings of the slow reaction
            nrcounter(r) = 0;
        end
    else
        PPnext(r) = log(1/rand(1));
        nrcounter(r) = 1;
    end
end

x(:,1) = x0;
j=0;

while t(end)<T
    j=j+1;
    
    % Wait times (in real time) to each next firing
    p = propfun(x(:,j));    
    rcWait = PPnext - rcT;
    rtWait = rcWait./p;
    
    % Find which reaction fires and when
    [tau_f,reaction] = min(rtWait);
    if tau_f == Inf % Catch a stalled process
        t(j+1) = T;
        x(:,j+1) = x(:,j);
        return
    end

    % Update real and reaction times
    t(j+1) = t(j)+tau_f;
    rcT = rcT + p.*tau_f;
    
    % Update state
    x(:,j+1) = x(:,j) + nu(:,reaction);
    
    % Update next reaction-clock firing time
    nrcounter(reaction) = nrcounter(reaction)+1;
    if slowflag(reaction)
        try
            PPnext(reaction) = slowPP{PPloc(reaction)}(nrcounter(reaction));
        catch ME
            PPnext(reaction) = Inf;
        end
    else
        PPnext(reaction) = PPnext(reaction) + log(1/rand(1)); % Keep making up the next fire time
    end
    
end

end