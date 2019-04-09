function [ESS,comptime,estimate] = ABC_realisation_properties(ABC_realisation, benchmark, parameterfun)

redSimIdx = ABC_realisation.benchmarkIdx;
fullSimIdx = ABC_realisation.benchmarkIdx(ABC_realisation.contFlags);

% Effective sample size
kw = ABC_realisation.kweights;
ESS = (sum(kw))^2 / sum((kw).^2);

% Computation time
c = benchmark.c;
delta = benchmark.delta;
comptime = sum(c(redSimIdx))+sum(delta(fullSimIdx));

% Estimates of everything
kproposals = benchmark.kproposals;
F_of_k = parameterfun(kproposals(:,redSimIdx));
estimate = sum(F_of_k .* kw,2) ./ sum(kw);

end