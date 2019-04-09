% Load the data the figures will be based on
load(strcat(dn,'/ABC_outputs'));
%mkdir(strcat(dn,'/figout'));

specieslist = {'mRNA 1','mRNA 2','mRNA 3','Protein 1','Protein 2','Protein 3'};

function_list = {'$P_{ABC}(n \in (1.9,2.1))$','$P_{ABC}(n \in (2.5,2.6))$','$E_{ABC}(n)$'};
num_functions = size(estimates_abc_post,1);

num_budgets = numel(budgets);


%% Fig 1: Reduction example
% An example of a coupled tau-leap / completion, randomly chosen

to = benchmark.to; xo = benchmark.xo; % Observed
tc = benchmark.tc; xc = benchmark.xc; % Coarse-grained
tf = benchmark.tf; xf = benchmark.xf; % Fine-grained
Fig1 = figure;
for j=1:6
    subplot(2,3,j);
    plot(tc,xc(j,:),'r--',tf,xf(j,:),'b-');
    title(specieslist{j},'FontWeight','normal');
    xlim([0 10]); ylim([0 1000]);
    if j>3
        xlabel('Time'); 
    end
    if rem(j,3)==1
        ylabel('Molecule Count');
    end
    axis square
end
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 8, 5], 'PaperUnits', 'Inches', 'PaperSize', [8 5])


%% Fig 2: Distances across benchmark
d = benchmark.d;
n = benchmark.kproposals(2,:);
Kh = benchmark.kproposals(5,:);

closeFlags = (d<epsilon);
matchIdx = (closeFlags(1,:)==closeFlags(2,:));
fpIdx = ((closeFlags(1,:)==1)&(closeFlags(2,:)==0));
fnIdx = ((closeFlags(1,:)==0)&(closeFlags(2,:)==1));

% Scatter d by tilde{d}
Fig2a = figure;
scatter(d(1,matchIdx),d(2,matchIdx),20,'filled');
hold on
scatter(d(1,fpIdx),d(2,fpIdx),20,'filled');
scatter(d(1,fnIdx),d(2,fnIdx),20,'filled');
plot(xlim,[epsilon epsilon],'k:');
plot([epsilon epsilon],ylim,'k:');

legend({'Matching estimator values','False positive','False negative','Threshold'},'Location','northwest')
xlabel('$\tilde d(\tilde{y},\tilde y_{obs})$','Interpreter','latex');
ylabel('$d(y,y_{obs})$','Interpreter','latex');
xticks(0:200:1000); yticks(0:200:1000);
title('Distance from data: low and high fidelity simulations','Interpreter','latex');
box on

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 5, 4], 'PaperUnits', 'Inches', 'PaperSize', [5 4])

% d by n
Fig2b = figure;
scatter(n(matchIdx),d(2,matchIdx),20,'filled');
hold on
scatter(n(fpIdx),d(2,fpIdx),20,'filled');
scatter(n(fnIdx),d(2,fnIdx),20,'filled');
plot(xlim,[epsilon epsilon],'k:');
%legend({'Matching','False Positive','False Negative','Threshold'},'Location','northwest')
xlabel('$n$','Interpreter','latex');
xticks(1:1:4); yticks(0:200:1000);
ylabel('$d(y,y_{obs})$','Interpreter','latex');
title('Distance from data: by $n$','Interpreter','latex');
box on
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 5, 4], 'PaperUnits', 'Inches', 'PaperSize', [5 4])

% d by Kh
Fig2c = figure;
scatter(Kh(matchIdx),d(2,matchIdx),20,'filled');
hold on
scatter(Kh(fpIdx),d(2,fpIdx),20,'filled');
scatter(Kh(fnIdx),d(2,fnIdx),20,'filled');
plot(xlim,[epsilon epsilon],'k:');
%legend({'Matching','False Positive','False Negative','Threshold'},'Location','northwest')
xlabel('$K_h$','Interpreter','latex');
ylabel('$d(y,y_{obs})$','Interpreter','latex');
title('Distance from data: by $K_h$','Interpreter','latex');
box on

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 5, 4], 'PaperUnits', 'Inches', 'PaperSize', [5 4])

%% Fig 3x: Estimator performance by budget
% Across all the functions of parameters considered, show the distribution
% of budget-constrained ABC estimates (together with the benchmark's 'real'
% estimate)

for j=1:num_functions
    Fig3{j} = figure;
    for k=1:num_budgets
        h{1,k} = subplot(num_budgets,4,4*k-3);
        histogram(estimates_abc_post_budget(j,:,1,k),'Normalization','pdf');
        yticklabels([]);
        ylabel(sprintf('Budget\n%.f min',budgets(k)/60));
        
        h{2,k} = subplot(num_budgets,4,4*k-2);
        histogram(estimates_ml_post_budget(j,:,j,k),'Normalization','pdf');
        yticklabels([]);
        
        h{3,k} = subplot(num_budgets,4,4*k-1);
        histogram(estimates_lz_post_budget(j,:,j,k),'Normalization','pdf'); % Match optimal eta to function
        yticklabels([]);

        h{4,k} = subplot(num_budgets,4,4*k-0);
        histogram(estimates_sl_post_budget(j,:,j,k),'Normalization','pdf');
        yticklabels([]);
    end
    xl = xlim(h{1,1});
    yl = ylim(h{4,end});
    for k=1:num_budgets
        xlim(h{1,k},xl); xlim(h{2,k},xl); xlim(h{3,k},xl); xlim(h{4,k},xl);
        ylim(h{1,k},yl); ylim(h{2,k},yl); ylim(h{3,k},yl); ylim(h{4,k},yl);
        xtext = xl(1)+0.05*(xl(2)-xl(1)); ytext = yl(1)+0.85*(yl(2)-yl(1));
        text(h{1,k},xtext,ytext,sprintf('$10^4 \\sigma^2 = %.2f$',(10^4)*var(estimates_abc_post_budget(j,:,1,k))),'Interpreter','latex');
        text(h{2,k},xtext,ytext,sprintf('$10^4 \\sigma^2 = %.2f$',(10^4)*var(estimates_ml_post_budget(j,:,j,k))),'Interpreter','latex');
        text(h{3,k},xtext,ytext,sprintf('$10^4 \\sigma^2 = %.2f$',(10^4)*var(estimates_lz_post_budget(j,:,j,k))),'Interpreter','latex');
        text(h{4,k},xtext,ytext,sprintf('$10^4 \\sigma^2 = %.2f$',(10^4)*var(estimates_sl_post_budget(j,:,j,k))),'Interpreter','latex');
        observed_variances(k,:,j) = [var(estimates_abc_post_budget(j,:,1,k)), ...
            var(estimates_ml_post_budget(j,:,j,k)), ...
            var(estimates_lz_post_budget(j,:,j,k)), ...
            var(estimates_sl_post_budget(j,:,j,k))];
    end
    title(h{1,1},sprintf('Rejection Sampling\n$\\eta_i = 1$'),'Interpreter','latex');
    title(h{2,1},sprintf('Multilevel\n$\\eta_1=\\eta_2$'),'Interpreter','latex');
    title(h{3,1},sprintf('Early Rejection\n$\\eta_1 = 1$'),'Interpreter','latex');
    title(h{4,1},sprintf('Multifidelity\n$\\eta_i = \\hat \\eta_i$'),'Interpreter','latex');
    
    spt = suptitle(string(function_list{j}));
    set(spt,'Interpreter','latex');
end

output_tables.theoretical_phi = [th_m2_abc_fn.*th_comptime_abc_fn diag(th_m2_ml_fn.*th_comptime_ml_fn) diag(th_m2_lz_fn.*th_comptime_lz_fn) diag(th_m2_sl_fn.*th_comptime_sl_fn)];
output_tables.theoretical_phi
(output_tables.theoretical_phi./output_tables.theoretical_phi(:,1))-1

%% Fig 4: Plot eta
spaceoffset = 0.02;
Fig4 = figure;
hold on;
scatter(eta_sl_ess(1),eta_sl_ess(2),'filled'); text(eta_sl_ess(1)+spaceoffset,eta_sl_ess(2)+spaceoffset,'Early accept/reject','Interpreter','latex'); 
scatter(eta_lz_ess(1),eta_lz_ess(2),'filled'); text(eta_lz_ess(1)-12.5*spaceoffset,eta_lz_ess(2)+2*spaceoffset,'Early rejection','Interpreter','latex')
scatter(eta_ml_ess(1),eta_ml_ess(2),'filled'); text(eta_ml_ess(1)+spaceoffset,eta_ml_ess(2)+spaceoffset,'Early decision','Interpreter','latex')
scatter(epp(1),epp(2),'filled'); text(epp(1)+spaceoffset,epp(2)+spaceoffset,'$\eta^{+/+}$','Interpreter','latex');
scatter(epm(1),epm(2),'filled'); text(epm(1)+spaceoffset,epm(2)-spaceoffset,'$\eta^{+/-}$','Interpreter','latex');
scatter(emp(1),emp(2),'filled'); text(emp(1)+spaceoffset,emp(2)+spaceoffset,'$\eta^{-/+}$','Interpreter','latex');
scatter(emm(1),emm(2),'filled'); text(emm(1)+spaceoffset,emm(2)-spaceoffset,'$\eta^{-/-}$','Interpreter','latex');
scatter(1,1,'filled'); text(1-12.5*spaceoffset,1+2*spaceoffset,'Rejection sampling','Interpreter','latex');

plot([1 1],[0 1],'k--');
plot([0 1],[0 1],'k--');

mesh = 0:0.01:1;
phi = getPhiValues(mesh,benchmark,epsilon);
contour(mesh,mesh,phi,min(min(phi))./[.99 .95 .9 .85 .8 .75 .6],'k:');
box on

title('Continuation probabilities and efficiency landscape','Interpreter','latex')
xlim([-0.1 1.1]); xlabel('$\eta_1$: continuation prob. if $\tilde y \in \tilde \Omega$','Interpreter','latex');
xticks(0:0.2:1); xtickformat('%.1f');
ylim([-0.1 1.1]); ylabel('$\eta_2$: continuation prob. if $\tilde y \notin \tilde \Omega$','Interpreter','latex');
yticks(0:0.2:1); ytickformat('%.1f');

%% Fig 5: Efficiency across fixed number of proposals: 

% Important eta efficiencies
eff_sl = ESS_sl_ess ./ comptime_sl_ess; % Effiency of multifidelity (optimal)
eff_lz = ESS_lz_ess ./ comptime_lz_ess; % Efficiency of lazy (optimal)
eff_ml = ESS_ml_ess ./ comptime_ml_ess; % Efficiency of lazy (optimal)
eff_abc = ESS_abc_post(1) ./ comptime_abc_post(1); % The benchmark efficiency

% Other eta efficiencies
eff_pp = ESS_pp ./ comptime_pp;
eff_pm = ESS_pm ./ comptime_pm;
eff_mp = ESS_mp ./ comptime_mp;
eff_mm = ESS_mm ./ comptime_mm;

% Fit normal distributions
d_sl = fitdist(eff_sl','normal');
d_lz = fitdist(eff_lz','normal');
d_ml = fitdist(eff_ml','normal');
d_pp = fitdist(eff_pp','normal');
d_pm = fitdist(eff_pm','normal');
d_mp = fitdist(eff_mp','normal');
d_mm = fitdist(eff_mm','normal');

% % Fig 5a: Sym Lazy vs Lazy vs ABC
% figure;
% xvals = 0:0.001:(1.4*max(eff_sl));
% plot(xvals, [pdf(d_sl,xvals); pdf(d_lz,xvals)],'LineWidth',2);
% hold on
% plot([eff_abc eff_abc],ylim,'k--','LineWidth',2);
% legend('Multifidelity','Lazy','Rejection');
% xlabel('Effective Samples per Second');
% yticklabels([]);
% title('Efficiency increases with laziness');

% Fig 5b: Sym Lazy with different eta
Fig5 = figure;
xvals = 0:0.001:(1.4*max(eff_sl));
plot(xvals, [pdf(d_sl,xvals); pdf(d_lz,xvals); pdf(d_ml,xvals)],'LineWidth',2)
hold on
plot(xvals, [pdf(d_pp,xvals); pdf(d_pm,xvals); pdf(d_mp,xvals); pdf(d_mm,xvals)],'LineWidth',2,'LineStyle',':');
plot([eff_abc eff_abc],ylim,'LineWidth',1.5);
legend({'Early accept/reject','Early rejection','Early decision',...
    '$\eta^{+/+}$','$\eta^{+/-}$','$\eta^{-/+}$','$\eta^{-/-}$','Rejection sampling'},...
    'Interpreter','latex');
xlabel('Effective samples per second','Interpreter','latex');
yticklabels([]);
xlim([0.1,0.25]);
xticklabels(0.1:0.05:0.25); xtickformat('%.2f');

box on

title('Distribution of efficiency across multiple realisations','Interpreter','latex');

% Empirical pairwise comparison of efficiencies
ESSF = [eff_sl' eff_lz' eff_ml' eff_mm' eff_pm' eff_pp' eff_mp' eff_abc*ones(size(eff_pp'))];

for a = 1:8
    for b = 1:8
        aa = ESSF(:,a); bb = ESSF(:,b);
        [aaa,bbb] = meshgrid(aa,bb);
        if a==b
            tables.eff_pw(a,b) = nan;
        else
            output_tables.efficiency_pairwise(a,b) = sum(sum(aaa>bbb))/numel(aaa);
        end
    end
end
output_tables.efficiency_pairwise

%% Fig 6: ESS by budget

% Pairwise comparison of ESS for sl and lz
p_sl_ESS_bigger_post = zeros(num_budgets,1);
%p_sl_ESS_bigger_pre = zeros(num_budgets,1);
for b=1:num_budgets
    [ESS_sl_post_mg, ESS_lz_post_mg] = meshgrid(ESS_sl_post_budget(:,1,b), ESS_lz_post_budget(:,1,b));
%    [ESS_sl_pre_mg, ESS_lz_pre_mg] = meshgrid(ESS_sl_pre_budget(:,b), ESS_lz_pre_budget(:,b));
    p_sl_ESS_bigger_post(b) = sum(sum(ESS_sl_post_mg > ESS_lz_post_mg)) / numel(ESS_sl_post_mg);
%    p_sl_ESS_bigger_pre(b) = sum(sum(ESS_sl_pre_mg > ESS_lz_pre_mg)) / numel(ESS_sl_pre_mg);
end

% Fig 6a: Known eta
Fig6 = figure;

yyaxis right
errorbar(budgets,mean(squeeze(ESS_sl_post_budget(:,1,:))),std(squeeze(ESS_sl_post_budget(:,1,:)))); 
hold on
errorbar(budgets,mean(squeeze(ESS_lz_post_budget(:,1,:))),std(squeeze(ESS_lz_post_budget(:,1,:))));
errorbar(budgets,mean(squeeze(ESS_ml_post_budget(:,1,:))),std(squeeze(ESS_ml_post_budget(:,1,:))));
errorbar(budgets,mean(squeeze(ESS_abc_post_budget)),std(squeeze(ESS_abc_post_budget)));
title('ESS from Fixed Computational Budget: Known Error');
db = diff(budgets);
xl = [min(budgets)-0.5*db(1), max(budgets)+0.5*db(end)];
xlim(xl)
xticks(budgets);
xlabel('Budget (s)')
ylabel('ESS')
ylim([0 1000]);

yyaxis left
plot(budgets,p_sl_ESS_bigger_post,'o','MarkerFaceColor',[0 0.4470 0.7410]);
ylabel('probability')
ylim([0.45 1]);

legend(...
    'Prob. symmetric ESS exceeds lazy ESS', ...
    'Multifidelity','Lazy','Multilevel',...
...    'Multifidelity (unknown error)','Lazy (unknown error)', ...
    'Rejection', ...
    'Location','northwest');



%% Save all

FolderName = strcat('figs');   % Your destination folder

saveas(Fig1,strcat(FolderName,'/Fig1'),'epsc');
saveas(Fig2a,strcat(FolderName,'/Fig2a'),'epsc');
saveas(Fig2b,strcat(FolderName,'/Fig2b'),'epsc');
saveas(Fig2c,strcat(FolderName,'/Fig2c'),'epsc');
for j=1:num_functions
saveas(Fig3{j},strcat(FolderName,'/Fig3-',num2str(j)),'epsc');
end
saveas(Fig4,strcat(FolderName,'/Fig4'),'epsc');
saveas(Fig5,strcat(FolderName,'/Fig5'),'epsc');
saveas(Fig6,strcat(FolderName,'/Fig6'),'epsc');

%% Function
function [phi] = getPhiValues(mesh,benchmark,epsilon)

path('../slabc',path);
[e1,e2] = meshgrid(mesh);
phi = zeros(size(e1));

for i1 = 1:size(e1,1)
    for i2 = 1:size(e1,2)
        [f1,f2] = theoretical_ESS([e1(i1,i2),e2(i1,i2)],benchmark,epsilon);
        phi(i1,i2) = f1*f2;
    end
end

end
