function [eta] = optimal_sl_eta(d,eps,c,delta,kproposals,lowerbound,parameterFun,contFlags)

if nargin<8
    contFlags = true(size(c));
end

closeFlags = (d<eps);
closeFlags = closeFlags(:,contFlags);

% If there's a function to estimate then optimal eta needs to minimise the
% variance of sum(wf)/sum(w) - else minimise variance of sum(w).
if nargin >= 7
    f_of_k = parameterFun(kproposals);
    numFun = size(f_of_k,1);
    f_of_k = f_of_k(:,contFlags);
    % Estimate ABC expectation of F using standard ABC (i.e. not any
    % yet-to-be-determined weights)
    mu = sum(f_of_k.*(closeFlags(2,:)==1),2) ./ sum(closeFlags(2,:)==1,2);
    Fweighting = (mu - f_of_k).^2;
else
    Fweighting = 1;
    numFun = 1;
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

% Ratios
Rp = p_fp ./ delta_p;
Rn = p_fn ./ delta_n;
R0 = (p_tp - p_fp) ./ c;
R1 = (p_tp + p_fn) ./ (c + delta_n + delta_p);

for nfun = 1:numFun
    
    % Optimal eta (fully lazy)
    if (Rp(nfun)<=R0(nfun)) && (Rn(nfun)<=R0(nfun))
        eta(nfun,:) = [sqrt(Rp(nfun)/R0(nfun)) sqrt(Rn(nfun)/R0(nfun))];
    elseif (Rp(nfun)>R1(nfun))&&(Rn(nfun)>R1(nfun))
        eta(nfun,:) = [1 1];
    elseif (Rp(nfun)>Rn(nfun))
        eta(nfun,:) = [1 sqrt(Rn(nfun)).*sqrt((c+delta_p)./p_tp(nfun))];
    else
        eta(nfun,:) = [sqrt(Rp(nfun)).*sqrt((c+delta_n)./p_fn(nfun)) 1];
    end
end

% Optimal eta (original lazy: force eta1 = 1)
% etalazy1(:,1) = ones(size(p_fp));
% etalazy1(:,2) = sqrt( p_fn ./ p_tp ) .* sqrt( (c+delta_p) ./ delta_n );

% Optimal eta (alternative lazy: force eta2 = 1)
% etalazy2(:,1) = sqrt( p_fp ./ (p_tp - p_fp + p_fn) ) .* sqrt( (c+delta_n) ./ delta_p );
% etalazy2(:,2) = ones(size(p_fn));


%for nfun = 1:numFun
%    if etahat(nfun,1)<1 && etahat(nfun,2)<1
%        eta(nfun,:) = etahat(nfun,:);
%    elseif etahat(1) > etahat(2)
%        eta(nfun,:) = etalazy1(nfun,:);
%    else
%        eta(nfun,:) = etalazy2(nfun,:);
%    end
%end

eta = max([lowerbound lowerbound],eta);
eta = min([1 1],eta);

end