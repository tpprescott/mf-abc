function [] = produce_ABC_realisations(dn,eps,num_realisations,parameterFun,budgets)
% Produces lots of ABC realisations from the benchmark data

%% LOAD ALL THE PARTS OF THE BENCHMARK

% Set up list
c = [];
delta = [];
d = [];
kproposals = [];

bmbits = dir(strcat(dn,'/output/*.mat'));
for fidx = 1:numel(bmbits)
    
    % Load the portion of the benchmark
    bmbit = load(strcat(dn,'/output/',bmbits(fidx).name));
    
    % How many outputs?
    bigi = max(bmbit.Nred);
    
    % Add them to the list
    c = [c bmbit.c(1:bigi)];
    delta = [delta bmbit.delta(1:bigi)];
    d = [d bmbit.d(:,1:bigi)];
    kproposals = [kproposals bmbit.kproposals(:,1:bigi)];
    
end

benchmark.c = c;
benchmark.delta = delta;
benchmark.d = d;
benchmark.kproposals = kproposals;
benchmark.tf = bmbit.tf;
benchmark.xf = bmbit.xf;
benchmark.tc = bmbit.tc;
benchmark.xc = bmbit.xc;
benchmark.to = bmbit.to;
benchmark.xo = bmbit.xo;

%% Calculate etas

% Posthoc estimates of best eta for each function
eta_sl_fn = optimal_sl_eta(d,eps,c,delta,kproposals,0.01,parameterFun);
eta_lz_fn = optimal_lz_eta(d,eps,c,delta,kproposals,0.01,parameterFun);
eta_ml_fn = optimal_ml_eta(d,eps,c,delta,kproposals,0.01,parameterFun);

num_functions = size(eta_sl_fn,1);
fun_chooser = eye(num_functions);

% Eta to optimise ESS
eta_sl_ess = optimal_sl_eta(d,eps,c,delta,kproposals,0.01);
eta_lz_ess = optimal_lz_eta(d,eps,c,delta,kproposals,0.01);
eta_ml_ess = optimal_ml_eta(d,eps,c,delta,kproposals,0.01);

% Box around optimal ESS
lambda = 0.5;
epp = lambda*eta_sl_ess + lambda*[1 1];
epm = lambda*eta_sl_ess + lambda*[1 0];
emp = lambda*eta_sl_ess + lambda*[0 1];
emm = lambda*eta_sl_ess + lambda*[0 0];

%% Produce a set of realisations of (sym) lazy ABC with no budget constraints
sl_post = cell(num_realisations,num_functions);
lz_post = cell(num_realisations,num_functions);
ml_post = cell(num_realisations,num_functions);
abc_post = cell(num_realisations,1);

for r=1:num_realisations
    for f=1:num_functions
        sl_post{r,f} = lABC_fixedeta(c,delta,d,eps,eta_sl_fn(f,:));
        lz_post{r,f} = lABC_fixedeta(c,delta,d,eps,eta_lz_fn(f,:));
        ml_post{r,f} = lABC_fixedeta(c,delta,d,eps,eta_ml_fn(f,:));
    end
    abc_post{r,1} = lABC_fixedeta(c,delta,d,eps,[1;1]);
end

%% Produce a set of realisations of (sym) lazy ABC using given budget constraints
sl_post_budget = cell(num_realisations,num_functions,numel(budgets));
lz_post_budget = cell(num_realisations,num_functions,numel(budgets));
ml_post_budget = cell(num_realisations,num_functions,numel(budgets));
abc_post_budget = cell(num_realisations,1,numel(budgets));
sl_ess_budget = cell(num_realisations,1,numel(budgets));
lz_ess_budget = cell(num_realisations,1,numel(budgets));
ml_ess_budget = cell(num_realisations,1,numel(budgets));

for b=1:numel(budgets)
    for r=1:num_realisations
        for f=1:num_functions
            sl_post_budget{r,f,b} = lABC_fixedeta(c,delta,d,eps,eta_sl_fn(f,:),budgets(b));
            lz_post_budget{r,f,b} = lABC_fixedeta(c,delta,d,eps,eta_lz_fn(f,:),budgets(b));
            ml_post_budget{r,f,b} = lABC_fixedeta(c,delta,d,eps,eta_ml_fn(f,:),budgets(b));
        end
        abc_post_budget{r,1,b} = lABC_fixedeta(c,delta,d,eps,[1;1],budgets(b)); % Include budgets on rejection ABC
        sl_ess_budget{r,1,b} = lABC_fixedeta(c,delta,d,eps,eta_sl_ess,budgets(b));
        lz_ess_budget{r,1,b} = lABC_fixedeta(c,delta,d,eps,eta_lz_ess,budgets(b));
        ml_ess_budget{r,1,b} = lABC_fixedeta(c,delta,d,eps,eta_ml_ess,budgets(b));
    end
end

% %% Produce a set of realisations of (sym) lazy ABC from unknown eta but targetting a particular function
% for r=1:num_realisations
%     for f=1:num_functions
%         Fn = @(x)(fun_chooser(f,:)*parameterFun(x));
%         sl_pre{r,f} = lABC_varyeta(kproposals,c,delta,d,eps,Fn,[0.01;0.01]);
%         lz_pre{r,f} = lABC_varyeta(kproposals,c,delta,d,eps,Fn,0.01);
%     end
% end
% 
% %% Produce a set of realisations of (sym) lazy ABC using given budget constraints from unknown eta
% 
% for b=1:numel(budgets)
%     for r=1:num_realisations
%         for f=1:num_functions
%             Fn = @(x)(fun_chooser(f,:)*parameterFun(x));
%             sl_pre_budget{r,f,b} = lABC_varyeta(kproposals,c,delta,d,eps,Fn,[0.01;0.01],budgets(b));
%             lz_pre_budget{r,f,b} = lABC_varyeta(kproposals,c,delta,d,eps,Fn,0.01,budgets(b));
%         end
%     end
% end

%% Consider ESS and other eta around
pp = cell(num_realisations,1);
pm = cell(num_realisations,1);
mp = cell(num_realisations,1);
mm = cell(num_realisations,1);
lz = cell(num_realisations,1);
sl = cell(num_realisations,1);
ml = cell(num_realisations,1);

for r=1:num_realisations
    pp{r,1} = lABC_fixedeta(c,delta,d,eps,epp);
    pm{r,1} = lABC_fixedeta(c,delta,d,eps,epm);
    mp{r,1} = lABC_fixedeta(c,delta,d,eps,emp);
    mm{r,1} = lABC_fixedeta(c,delta,d,eps,emm);
    lz{r,1} = lABC_fixedeta(c,delta,d,eps,eta_lz_ess);
    sl{r,1} = lABC_fixedeta(c,delta,d,eps,eta_sl_ess);
    ml{r,1} = lABC_fixedeta(c,delta,d,eps,eta_ml_ess);
end

%% SAVE IT ALL
save(strcat(dn,'/ABC_realisations'))