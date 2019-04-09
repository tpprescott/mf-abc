function [moment2nd, comptime] = theoretical_ESS(eta,benchmark,eps,parameterFun)
%Theoretical ESS: Calculate the value being minimised by the choice of eta

closeFlags = (benchmark.d<eps);

if nargin<4
    Fweighting = 1;
else
    f_of_k = parameterFun(benchmark.kproposals);
    mu = sum(f_of_k.*(closeFlags(2,:)==1),2) ./ sum(closeFlags(2,:)==1,2);
    Fweighting = (mu - f_of_k).^2;
end

% Put into the expectations
p_tp = mean(Fweighting.*((closeFlags(1,:)==1).*(closeFlags(2,:)==1)),2);
p_fp = mean(Fweighting.*((closeFlags(1,:)==1).*(closeFlags(2,:)==0)),2);
p_fn = mean(Fweighting.*((closeFlags(1,:)==0).*(closeFlags(2,:)==1)),2);

% Estimate computation times
c = mean(benchmark.c);
delta_p = mean(benchmark.delta .* (closeFlags(1,:)==1));
delta_n = mean(benchmark.delta .* (closeFlags(1,:)==0));

moment2nd = (p_tp - p_fp) + p_fp/eta(1) + p_fn/eta(2);
comptime = c + eta(1)*delta_p + eta(2)*delta_n;

end

