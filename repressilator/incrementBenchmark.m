function [dn] = incrementBenchmark(tau,new_simulation_number,num_ABC_realisations,epsilon,budgets)

T = 10;
x0 = [0;0;0;40;20;60];
k_o = [1;2;5;1000;20];
dn = getTauFolder(tau);

%% Generate some new simulations if needed
if new_simulation_number>0
    produceBenchmarkData(dn,tau,new_simulation_number,T,x0,k_o);
end

%% Produce some ABC realisations from the benchmark
% Use the entire benchmark but randomly reordered
% num_realisations times. F points to the various functions we want to
% estimate (each of which is going to have its own optimal eta)
path('../mfabc',path);
F = @parameterFun; % Specified at bottom of this file

% Convert simulations at both fidelities into realisations of ABC algorithm
produce_ABC_realisations(dn,epsilon,num_ABC_realisations,F,budgets);
load(strcat(dn,'/ABC_realisations.mat'));

%% Return properties of realisations that can be plotted

% No budget
for r=1:num_ABC_realisations
    for f=1:num_functions
        % Posthoc analysis (known best eta)
        [ESS_sl_post(r,f),comptime_sl_post(r,f),estimates_sl_post(:,r,f)] = ABC_realisation_properties(sl_post{r,f},benchmark,F);
        [ESS_lz_post(r,f),comptime_lz_post(r,f),estimates_lz_post(:,r,f)] = ABC_realisation_properties(lz_post{r,f},benchmark,F);
        [ESS_ml_post(r,f),comptime_ml_post(r,f),estimates_ml_post(:,r,f)] = ABC_realisation_properties(ml_post{r,f},benchmark,F);
        
%         % Prehoc analysis (estimate eta as you go)
%         [ESS_sl_pre(r,f),comptime_sl_pre(r,f),estimates_sl_pre(:,r,f)] = ABC_realisation_properties(sl_pre{r,f},F);
%         [ESS_lz_pre(r,f),comptime_lz_pre(r,f),estimates_lz_pre(:,r,f)] = ABC_realisation_properties(lz_pre{r,f},F);
%         
%         eta_est_sl_pre(:,r,f) = sl_pre{r,f}.eta(:,end);
%         eta_est_lz_pre(:,r,f) = lz_pre{r,f}.eta(:,end);
        
    end
    
    % Other etas
    [ESS_abc_post(r),comptime_abc_post(r),estimates_abc_post(:,r)] = ABC_realisation_properties(abc_post{r},benchmark,F);
    
    [ESS_pp(r),comptime_pp(r),estimates_pp(:,r)] = ABC_realisation_properties(pp{r},benchmark,F);
    [ESS_pm(r),comptime_pm(r),estimates_pm(:,r)] = ABC_realisation_properties(pm{r},benchmark,F);
    [ESS_mp(r),comptime_mp(r),estimates_mp(:,r)] = ABC_realisation_properties(mp{r},benchmark,F);
    [ESS_mm(r),comptime_mm(r),estimates_mm(:,r)] = ABC_realisation_properties(mm{r},benchmark,F);
    [ESS_sl_ess(r),comptime_sl_ess(r),estimates_sl_ess(:,r)] = ABC_realisation_properties(sl{r},benchmark,F);
    [ESS_lz_ess(r),comptime_lz_ess(r),estimates_lz_ess(:,r)] = ABC_realisation_properties(lz{r},benchmark,F);
    [ESS_ml_ess(r),comptime_ml_ess(r),estimates_ml_ess(:,r)] = ABC_realisation_properties(ml{r},benchmark,F);

end

% Budget
for r=1:num_ABC_realisations
    for b=1:numel(budgets)
        for f=1:num_functions
            
            % Posthoc analysis (known best eta)
            [ESS_sl_post_budget(r,f,b),comptime_sl_post_budget(r,f,b),estimates_sl_post_budget(:,r,f,b)] = ABC_realisation_properties(sl_post_budget{r,f,b},benchmark,F);
            [ESS_lz_post_budget(r,f,b),comptime_lz_post_budget(r,f,b),estimates_lz_post_budget(:,r,f,b)] = ABC_realisation_properties(lz_post_budget{r,f,b},benchmark,F);
            [ESS_ml_post_budget(r,f,b),comptime_ml_post_budget(r,f,b),estimates_ml_post_budget(:,r,f,b)] = ABC_realisation_properties(ml_post_budget{r,f,b},benchmark,F);
            
%             % Prehoc analysis (estimate eta as you go)
%             [ESS_sl_pre_budget(r,f,b),comptime_sl_pre_budget(r,f,b),estimates_sl_pre_budget(:,r,f,b)] = ABC_realisation_properties(sl_pre_budget{r,f,b},F);
%             [ESS_lz_pre_budget(r,f,b),comptime_lz_pre_budget(r,f,b),estimates_lz_pre_budget(:,r,f,b)] = ABC_realisation_properties(lz_pre_budget{r,f,b},F);
%             
%             eta_est_sl_pre_budget(:,r,f,b) = sl_pre_budget{r,f,b}.eta(:,end);
%             eta_est_lz_pre_budget(:,r,f,b) = lz_pre_budget{r,f,b}.eta(:,end);
%             
        end
        [ESS_abc_post_budget(r,1,b),comptime_abc_post_budget(r,1,b),estimates_abc_post_budget(:,r,1,b)] = ABC_realisation_properties(abc_post_budget{r,1,b},benchmark,F);
        [ESS_sl_ess_budget(r,1,b),comptime_sl_ess_budget(r,1,b),estimates_sl_ess_budget(:,r,1,b)] = ABC_realisation_properties(sl_ess_budget{r,1,b},benchmark,F);
        [ESS_lz_ess_budget(r,1,b),comptime_lz_ess_budget(r,1,b),estimates_lz_ess_budget(:,r,1,b)] = ABC_realisation_properties(lz_ess_budget{r,1,b},benchmark,F);
        [ESS_ml_ess_budget(r,1,b),comptime_ml_ess_budget(r,1,b),estimates_ml_ess_budget(:,r,1,b)] = ABC_realisation_properties(ml_ess_budget{r,1,b},benchmark,F);
    end
end

%% Theoretical functions being minimised

% ESS first
[th_m2_sl_ess,th_comptime_sl_ess] = theoretical_ESS(eta_sl_ess,benchmark,epsilon);
[th_m2_lz_ess,th_comptime_lz_ess] = theoretical_ESS(eta_lz_ess,benchmark,epsilon);
[th_m2_ml_ess,th_comptime_ml_ess] = theoretical_ESS(eta_ml_ess,benchmark,epsilon);
[th_m2_abc_ess,th_comptime_abc_ess] = theoretical_ESS([1;1],benchmark,epsilon);

% Estimator functions too: the diagonal elements are the optima
for f=1:num_functions
[th_m2_sl_fn(:,f),th_comptime_sl_fn(:,f)] = theoretical_ESS(eta_sl_fn(f,:),benchmark,epsilon,F);
[th_m2_lz_fn(:,f),th_comptime_lz_fn(:,f)] = theoretical_ESS(eta_lz_fn(f,:),benchmark,epsilon,F);
[th_m2_ml_fn(:,f),th_comptime_ml_fn(:,f)] = theoretical_ESS(eta_ml_fn(f,:),benchmark,epsilon,F);
end
[th_m2_abc_fn,th_comptime_abc_fn] = theoretical_ESS([1;1],benchmark,epsilon,F);

%% Save what needs to be plotted
% Don't bother saving all the realisations though
sizeBenchmark = numel(benchmark.c);
save(strcat(dn,'/ABC_outputs'),'ESS*','comptime*','estimates*','eta*','th*'...
    ,'benchmark','epsilon', 'budgets','emm','emp','epp','epm');


end

%% Function(s) of parameter to estimate
function f = parameterFun(k)

f = [(k(2,:)<2.1).*(k(2,:)>1.9);
    (k(2,:)<2.6).*(k(2,:)>2.5);
    k(2,:)];
end