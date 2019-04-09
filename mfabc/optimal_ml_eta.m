function [eta] = optimal_ml_eta(d,eps,c,delta,kproposals,lowerbound,parameterFun,contFlags)

if nargin<8
    contFlags = true(size(c));
end

closeFlags = (d<eps);
closeFlags = closeFlags(:,contFlags);

% Function, if required
if nargin >= 7
    f_of_k = parameterFun(kproposals);
    f_of_k = f_of_k(:,contFlags);
    % Estimate ABC expectation of F using standard ABC (i.e. not any
    % yet-to-be-determined weights)
    mu = sum(f_of_k.*(closeFlags(2,:)==1),2) ./ sum(closeFlags(2,:)==1,2);
    Fweighting = (mu - f_of_k).^2;
else
    Fweighting = 1;
end

% Put into the expectations
p_tp = mean(Fweighting.*((closeFlags(1,:)==1).*(closeFlags(2,:)==1)),2);
p_fp = mean(Fweighting.*((closeFlags(1,:)==1).*(closeFlags(2,:)==0)),2);
p_fn = mean(Fweighting.*((closeFlags(1,:)==0).*(closeFlags(2,:)==1)),2);

% Estimate computation times
c = mean(c);
delta = delta(:,contFlags);
delta_p = mean(delta .* (closeFlags(1,:)==1));
delta_n = mean(delta .* (closeFlags(1,:)==0));

% Optimal eta (standard multilevel: force eta1 = eta2)
eta_common = sqrt(c / (delta_p + delta_n)) * sqrt((p_fp + p_fn)./(p_tp - p_fp));

% Need to code in the optimum when one or other is not in [0,1]; revert
% to an asymmetric case.
eta = max([lowerbound lowerbound],[eta_common eta_common]);
eta = min([1 1],eta);

end