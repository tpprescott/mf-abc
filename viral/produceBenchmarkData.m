function [] = produceBenchmarkData(num_trials, npop, T, x0, k_o)
% Produce a benchmark dataset: one observed output, and ntrials reductions and
% their completions.

% What to save as
[~,hn]=system('hostname');
hn=extractBefore(hn,'.');
dt=datestr(datetime('now'),'mmdd-HHMM-');
fn=strcat(dt,hn);
fn_temp = strcat('output/',fn,'-temp');
fn_complete = strcat('output/',fn,'-complete');

% Add simulations to path
path('../simulation',path);

%% INITIALISE SIMULATIONS
% Stoichiometry
S = [0 1 0 0;
    1 -1 0 0;
    0 0 1 0;
    -1 0 0 0;
    0 0 -1 0;
    0 -1 -1 1];
S = S';

% Reduction: only stochastically simulate a subset of reactions
redreactionIdx = [1 2 4 6];
redS = S(:,redreactionIdx);

% Times of interest
to = 1:T; tc = 1:T; tf = 1:T;

%% GENERATE SYNTHETIC DATA AS A CELL POPULATION USING FULL-SCALE SIMULATION

propfun_nominal = @(x)fullpropensity(x,k_o);
rng(1);
[xo, ~] = fullsimpop(T,x0,npop,S,propfun_nominal);

rng('shuffle');

%% Initialise dataset
kproposals = zeros(numel(k_o),num_trials);
c = zeros(1,num_trials);
delta = zeros(1,num_trials);
dist_r = zeros(2,num_trials);
d = zeros(2,num_trials);
% Record outputs
%vc_rec = zeros(npop,T,num_trials);
%vf_rec = zeros(npop,T,num_trials);


%% Generate dataset
ntrials_inc = 10;
for n = 1:num_trials
    
    %% Select parameter vector from prior
    k = kprior(k_o); % Base prior distribution on the nominal that generates data.
    kproposals(:,n) = k;
    
    % Simulate population with randomly selected parameters
    propfun = @(x)fullpropensity(x,k);
    redpropfun = @(x)redpropensity(x,k,redreactionIdx);
    xcorrfun = @(x)xcorrection(x,k);
    
    %% Simulate reduced system first, and time
    [xc, simtimes_c, PPpop_c] = redsimpop(T,x0,npop,redS,redpropfun,xcorrfun);
    [dist_r(1,n),d(1,n)] = output_distance(xo,xc,T);
    c(1,n) = sum(simtimes_c);
    
    
    %% Complete the simulation and time
    [xf, simtimes_f] = completesimpop(T,x0,PPpop_c,propfun,S,redreactionIdx);
    [dist_r(2,n), d(2,n)] = output_distance(xo, xf, T);
    delta(1,n) = sum(simtimes_f);
    
    %% Record outputs
%    vc_rec(:,:,n) = xc; % Coarse
%    vf_rec(:,:,n) = xf; % Fine
    
    if mod(n,ntrials_inc)==0
        Nred = n; clear PPpop_c
        save(char(fn_temp));
    end
end
Nred = n; clear PPpop_c
save(char(fn_complete));
delete(char(strcat(fn_temp,'.mat')));

end

%% Functions

function v = fullpropensity(x,k)

v = zeros(numel(k), size(x,2));

v(1,:) = x(1,:);
v(2,:) = x(2,:);
v(3,:) = x(1,:);
v(4,:) = x(1,:);
v(5,:) = x(3,:);
v(6,:) = x(2,:) .* x(3,:);

v = k.*v;

end

function v = redpropensity(x,k,reactionIdx)

v = fullpropensity(x,k);
v = v(reactionIdx,:);

end

function [tau_f, xcorrval] = xcorrection(x,k)
% Function that inputs a wait time (for the next stochastic reaction) and
% the state at the start of the wait time, and returns a deterministic
% struct value at the end of the wait time.

try
    err = (k(3)/k(5))*x(1) - x(3);
    tau_f = (-1/k(5)) * reallog( 1 - abs(1/err)); % Wait time
catch ME
    tau_f = Inf;
end

xcorrval = [0; 0; sign(err); 0]; % plus or minus 1 to struct depending on the values of struct and template

end

function k = kprior(ko)
k=ko;
rn = -1 + 2*rand(1); % random number between -1 and 1
% rn([4,5]) = [0;0]; % No uncertainty in degradation rates
rf = 1.5 .^ rn; % random factors between 2/3 and 3/2: log-uniform distribution
k(1) = ko(1).*rf;

end

function [dist_recover, dist_virus] = output_distance(vout1,vout2,T)

N1 = size(vout1,1);
N2 = size(vout2,1);

% Compare proportion of uninfected cells
r1 = sum(sum(vout1,2)<5);
r2 = sum(sum(vout2,2)<5);
if (r1==N1)||(r2==N2)
    dist_recover = Inf;
    dist_virus = Inf;
    return
end
rr1 = r1/N1; rr2 = r2/N2;
dist_recover = abs(rr1-rr2);

% Need to compare viral outputs from each infected cell: ensure that
% that the distance is independent of infected population size and
% experiment length.
v1 = sum(vout1,1)/(N1-r1);
v2 = sum(vout2,1)/(N2-r2);
dist_virus = sqrt(sum((v1 - v2).^2)/T);
end
