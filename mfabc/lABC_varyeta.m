function [ABCoutput] = lABC_varyeta(kproposals,c,delta,d,eps,Fn,change_rate,budget)
% A random ABC output using a fixed pair of eta = (eta1,eta2) where eta1 is
% the continuation probability for epsilon-close reductions, and eta2 is
% the continuation probability for epsilon-far reductions.
% Vary the input numbers: if budget then ensure output is bounded above by
% a given budget, else use the entire benchmark dataset. If eta_rate then
% allow evolution from eta to a parallel estimate of the optimal eta.

%% Lazy or symmetric lazy
if numel(change_rate) == 2
    optimal_eta_vary = @optimal_sl_eta_vary;
else
    optimal_eta_vary = @optimal_lz_eta_vary;
    change_rate = [0;change_rate];
end


%% Randomly reorder the input vector
N = numel(c);
sidx = randperm(N);
c = c(:,sidx);
delta = delta(:,sidx);
kproposals = kproposals(:,sidx);
d = d(:,sidx);

%% Work out what has fallen in the threshold
closeFlags = (d<eps);

%% Define continuation probabilities in parallel with continuation decision
u = rand(size(c));
eta = ones(size(d));
contProbs = ones(size(c));
contProbs_prev = zeros(size(c));

c = cumsum(c)./(1:N);

% Some cumulative information depends on continuation
% Need to loop to take into account filtered information
% Information all comes from indices before the one in question
while sum(contProbs_prev ~= contProbs)>0
    contFlags = (u<contProbs);
    
    V_tp = cumsum((Fn(kproposals).^2) .* ((closeFlags(1,:)==1).*(closeFlags(2,:)==1)) .* (contFlags==1)) ./ cumsum(contFlags);
    V_fp = cumsum((Fn(kproposals).^2) .* ((closeFlags(1,:)==1).*(closeFlags(2,:)==0)) .* (contFlags==1)) ./ cumsum(contFlags);
    V_fn = cumsum((Fn(kproposals).^2) .* ((closeFlags(1,:)==0).*(closeFlags(2,:)==1)) .* (contFlags==1)) ./ cumsum(contFlags);
    
    delta_p = cumsum(delta.*contFlags.*(closeFlags(1,:)==1))./cumsum(contFlags);
    delta_n = cumsum(delta.*contFlags.*(closeFlags(1,:)==0))./cumsum(contFlags);
    
    % Optimal eta given here
    etahat = optimal_eta_vary(V_tp,V_fp,V_fn,c,delta_p,delta_n);
    
    % Actual eta changes slowly: and doesn't change at all until at least
    % 30 true positives found.
    n_tp = cumsum((closeFlags(1,:)==1).*(closeFlags(2,:)==1).*contFlags); n0 = 1+sum(n_tp<30);
    
    % Solution to linear system eta(n+1) = eta(n) + lambda(etahat(n+1) - eta(n))
    eta_vary_transient = (1-change_rate).^[1:N-n0];
    eta_vary_input_1 = change_rate(1)*conv(etahat(1,n0+1:end),(1-change_rate(1)).^[0:N-n0-1]);
    eta_vary_input_2 = change_rate(2)*conv(etahat(2,n0+1:end),(1-change_rate(2)).^[0:N-n0-1]);
    eta_vary = eta_vary_transient + [eta_vary_input_1(1:N-n0); eta_vary_input_2(1:N-n0)];
    eta = [ones(2,n0) eta_vary];
    
    contProbs_prev = contProbs;
    contProbs = closeFlags(1,:).*eta(1,:) + (1-closeFlags(1,:)).*eta(2,:);
end

% Continuation probabilities given by eta
mismatchw = 1./contProbs;
kweights = closeFlags(1,:) + (closeFlags(2,:)-closeFlags(1,:)).*mismatchw.*contFlags;

%% Apply budget
cumCompTime = cumsum(c + contFlags.*delta);
if nargin<8
    budget = Inf;
end
incFlag = (cumCompTime < budget);

ABCoutput.kproposals = kproposals(:,incFlag);
ABCoutput.closeFlags = closeFlags(:,incFlag);
ABCoutput.contFlags = contFlags(:,incFlag);
ABCoutput.kweights = kweights(:,incFlag);
ABCoutput.c = c(:,incFlag);
ABCoutput.delta = contFlags(:,incFlag).*delta(:,incFlag);
ABCoutput.eta = eta(:,incFlag);

end

function eta = optimal_sl_eta_vary(V_tp,V_fp,V_fn,c,delta_p,delta_n)
% Get optimal eta (which varies along dimension 2 with the input estimates)

if V_tp<V_fp
    eta=[1;1];
    return
end

etahat = [ sqrt( V_fp ./ (V_tp - V_fp) ) .* sqrt( c ./ delta_p );
    sqrt( V_fn ./ (V_tp - V_fp) ) .* sqrt( c ./ delta_n )];

eta = [[1;1] etahat(:,1:end-1)];
eta = max([0.01;0.01],eta);
eta = min([1;1],eta);

end

function eta = optimal_lz_eta_vary(V_tp,V_fp,V_fn,c,delta_p,delta_n)

etahat = [ones(size(c));
    sqrt( V_fn ./ V_tp ) .* sqrt( (c+delta_p) ./ delta_n )];

eta = [[1;1] etahat(:,1:end-1)];
eta = max([0.01;0.01],eta);
eta = min([1;1],eta);

end