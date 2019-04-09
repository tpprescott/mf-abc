function [dn] = produceBenchmarkData(dn, tau_leap, Nred, T, x0, k_o)
% Nred = Total number of two-stage simulations. T = time interval length.
% x0 = i.c. k_o = nominal parameters.

mkdir(strcat(dn,'/output'));

d8 = datestr(now(),'mm-dd-HH-MM');
fn = strcat(d8,'-bmbit');

path('../simulation',path);

%% Repressilator set up:
% --> x_i -->  for i=1,2,3
% --> X_i -->  for i=1,2,3
% X_j inhibits production of x_i for i=1,2,3 and j=3,1,2

nu = [eye(3) -eye(3) zeros(3) zeros(3);
    zeros(3) zeros(3) eye(3) -eye(3)];
% Propensity function defined at the end.


%% Generate fixed "real" data
rng(3);

p = @(x)propensity(x,k_o);
summarytimes = 0:1:T;

tic;
% Simulate
[to,xo] = gillespieDM(T,x0,nu,p);
% Generate summary stats
so = summaryStat(to,xo,summarytimes);
DM_time = toc;

rng('shuffle')

%% Two-stage simulations
c = zeros(1,Nred);
delta = zeros(1,Nred);
d = zeros(2,Nred);
kproposals = zeros(numel(k_o),Nred);

for n=1:Nred
    sprintf('Sim: %.3f',n/Nred)
    
    k = kprior(k_o); % This is where k would be selected from its prior or a proposal based on MCMC
    
    p = @(x)propensity(x,k); % Create handle to pass into simulations
    dp = @(x)diff_propensity(x,k);
    
    % Tau leap simulation
    tic;
    [tc,xc,partialPP] = tauleapsim_noneg(T,x0,nu,p,dp,tau_leap,5,0.1);
    sc = summaryStat(tc,xc,summarytimes);
    c(1,n) = toc; % Time each simulation, to optimise ESS per simulation time
    
    % Measure distance between summary stats
    d(1,n) = measureDistance(so,sc,summarytimes);
    
    kproposals(:,n) = k;
    
    % Weight based on full-scale simulation
    tic;
    % 1. Bridge underlying PP from coarse description to fine.
    % Note that the coarse description also has some fine description
    % too
    for i = 1:size(nu,2)
        bridgedPP{i} = PPBridge_incexact(partialPP.IntervalEventCounts(i,:),partialPP.IntervalLengths(i,:),partialPP.exactSteps{i});
    end
    % 2. Map PP to Gillespie trajectory
    [tf,xf] = gillespiePP(T,x0,nu,p,bridgedPP);
    
    % Summary statistic
    sf = summaryStat(tf,xf,summarytimes);
    delta(1,n) = toc;
    
    % Measure distance between summary stats
    d(2,n) = measureDistance(so,sf,summarytimes);
    
    if 1==0 % eps_t_close*eps_close
        %% Plot a reduced/completed trajectory pair
        figure;
        for statedim = 1:6
            subplot(2,3,statedim), plot(tc,xc(statedim,:),'r-',tf,xf(statedim,:),'b:',to,xo(statedim,:),'k--')
            legend('tau-leap','stochastic','observed')
            xlim([0 T]);
        end
    end

end

save(strcat(dn,'/output/',fn), 'kproposals', 'c', 'delta', 'd', 'Nred', 'to', 'xo', 'tc', 'xc', 'tf', 'xf');

end


%% Subfunctions

% Propensity function

function p = propensity(x,k)
% k = (alpha0, n, beta, alpha)

p = [k(1) + k(4).*(k(5).^k(2))./((k(5).^k(2)) + (x([6 4 5],:).^k(2))); % transcription of m_i inhibited by p_{i-1}
    x(1:3,:); % degradation of m_i
    k(3).*x(1:3,:); % translation of p_i from m_i
    k(3).*x(4:6,:)]; % degradation of p_i

p = p.*(p>=0);

end

function dp = diff_propensity(x,k)

dp = zeros(12,6);

dp(1,6) = -k(4)*k(2).*(k(5).^k(2)).*(x(6,:).^(k(2)-1))*(k(5).^k(2) + x(6,:).^k(2)).^(-2);
dp(2,4) = -k(4)*k(2).*(k(5).^k(2)).*(x(4,:).^(k(2)-1))*(k(5).^k(2) + x(4,:).^k(2)).^(-2);
dp(3,5) = -k(4)*k(2).*(k(5).^k(2)).*(x(5,:).^(k(2)-1))*(k(5).^k(2) + x(5,:).^k(2)).^(-2);
dp(4:6,1:3) = eye(3);
dp(7:9,1:3) = k(3)*eye(3);
dp(10:12,4:6) = k(3)*eye(3);

end

% Draw from prior for k
function k = kprior(nominal)
% k = [alpha_0, n, beta, alpha, Kh]

%u = rand(size(nominal));
%k = [0;0;0;500] + [10;10;20;2000].*u;

k = nominal;
% Vary:
% n between 1 and 4
% Kh between 10 and 30
k([2,5]) = [1;10] + [3;20].*rand(2,1);

end

% Generate summary statistics from simulation data: vectors at given time
% points
function s = summaryStat(t,x,ts)

s = interp1(t,x',ts);
s = s';

end

% Compare summary stats
function [d] = measureDistance(s1,s2,t)

dt = diff(t);

e = (s1 - s2).^2;
ee = 0.5*(e(:,1:end-1) + e(:,2:end));
eeint = sum(ee.*dt,2)/(max(t)-min(t));
d = sqrt(sum(eeint));

end

