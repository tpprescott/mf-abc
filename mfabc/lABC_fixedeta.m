function [ABCoutput] = lABC_fixedeta(c,delta,d,eps,fixed_eta,budget)
% A random ABC output using a fixed pair of eta = (eta1,eta2) where eta1 is
% the continuation probability for epsilon-close reductions, and eta2 is
% the continuation probability for epsilon-far reductions.
% Vary the input numbers: if budget then ensure output is bounded above by
% a given budget, else use the entire benchmark dataset. If eta_rate then
% allow evolution from eta to a parallel estimate of the optimal eta.

%% Randomly reorder the input vector
sidx = randperm(numel(c));
c = c(:,sidx);
delta = delta(:,sidx);
d = d(:,sidx);

%% Work out what has fallen in the threshold
closeFlags = (d<eps);

%% Define continuation probabilities and randomly decide whether to continue
contprobs = closeFlags(1,:)*fixed_eta(1) + (1-closeFlags(1,:))*fixed_eta(2);
contFlags = (rand(size(contprobs)) < contprobs);
mismatchw = 1./contprobs;

kweights = closeFlags(1,:) + (closeFlags(2,:)-closeFlags(1,:)).*mismatchw.*contFlags;

%% Apply budget
cumCompTime = cumsum(c + contFlags.*delta);
if nargin<6
    budget = Inf;
end
incFlag = (cumCompTime < budget);

ABCoutput.kweights = kweights(:,incFlag);
ABCoutput.benchmarkIdx = sidx(:,incFlag);
ABCoutput.contFlags = contFlags(:,incFlag);

end