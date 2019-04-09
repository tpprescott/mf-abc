%% Fig 1: Reduction example
% An example of a coupled tau-leap / completion, randomly chosen
rng(1);
singlesim;
Fig1 = gcf;
set(Fig1, 'Units', 'Inches', 'Position', [0, 0, 10, 4], 'PaperUnits', 'Inches', 'PaperSize', [10 4])

%% Load the data the figures will be based on
load('ABC_outputs');
path('../slabc',path);

%% Fig 2: Distances across benchmark
d = benchmark.d;
k1 = benchmark.kproposals(1,:);

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
title('Distance from data: low and high-fidelity simulations','Interpreter','latex');
box on

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 5, 4], 'PaperUnits', 'Inches', 'PaperSize', [5 4])

% create a new pair of axes inside current figure
insetfig = axes('position',[.65 .175 .25 .25]);
box on % put box around new pair of axes

min_d1 = min(min(d(1,fpIdx)),min(d(1,fnIdx)));
min_d2 = min(min(d(2,fpIdx)),min(d(2,fnIdx)));
max_d1 = max(max(d(1,fpIdx)),max(d(1,fnIdx)));
max_d2 = max(max(d(2,fpIdx)),max(d(2,fnIdx)));
half_boxwidth = max(abs([min_d1 min_d2 max_d1 max_d2] - epsilon));
dmatch = d(:,matchIdx); 
dmatchzoom = dmatch(:,dmatch(1,:)<epsilon+half_boxwidth & dmatch(1,:)>epsilon-half_boxwidth & dmatch(2,:)<epsilon+half_boxwidth & dmatch(2,:)>epsilon-half_boxwidth);

hold on
scatter(dmatchzoom(1,:),dmatchzoom(2,:),10,'filled');
scatter(d(1,fpIdx),d(2,fpIdx),10,'filled');
scatter(d(1,fnIdx),d(2,fnIdx),10,'filled');
plot([epsilon-half_boxwidth, epsilon+half_boxwidth],[epsilon epsilon],'k:');
plot([epsilon epsilon],[epsilon-half_boxwidth, epsilon+half_boxwidth],'k:');
xticks([2.2 2.5 2.8]);
yticks([2.2 2.5 2.8]);
axis tight


% d by k1
Fig2b = figure;
scatter(k1(matchIdx),d(2,matchIdx),20,'filled');
hold on
scatter(k1(fpIdx),d(2,fpIdx),20,'filled');
scatter(k1(fnIdx),d(2,fnIdx),20,'filled');
xlim([2/3,3/2]);
plot(xlim,[epsilon epsilon],'k:');
%legend({'Matching','False Positive','False Negative','Threshold'},'Location','northwest')
xlabel('$k_1$','Interpreter','latex');
xticks(0.7:0.1:1.4); xtickformat('%.1f')
ylabel('$d(y,y_{obs})$','Interpreter','latex');
title('Distance from data: by $k_1$','Interpreter','latex');
box on

set(gcf, 'Units', 'Inches', 'Position', [0, 0, 5, 4], 'PaperUnits', 'Inches', 'PaperSize', [5 4])

%% Fig 4: Plot eta
spaceoffset = 0.02;
Fig4 = figure;
hold on;
scatter(eta_sl_ess(1),eta_sl_ess(2),'filled'); text(eta_sl_ess(1)+spaceoffset,eta_sl_ess(2)+spaceoffset,'Symmetric Lazy'); 
scatter(eta_lz_ess(1),eta_lz_ess(2),'filled'); text(eta_lz_ess(1)+spaceoffset,eta_lz_ess(2)+spaceoffset,'Lazy')
scatter(eta_ml_ess(1),eta_ml_ess(2),'filled'); text(eta_ml_ess(1)-spaceoffset,eta_ml_ess(2)+3*spaceoffset,'Multilevel')
%scatter(epp(1),epp(2),'filled'); text(epp(1)+spaceoffset,epp(2)+spaceoffset,'+/+');
%scatter(epm(1),epm(2),'filled'); text(epm(1)+spaceoffset,epm(2)+spaceoffset,'+/-');
%scatter(emp(1),emp(2),'filled'); text(emp(1)+spaceoffset,emp(2)+spaceoffset,'-/+');
%scatter(emm(1),emm(2),'filled'); text(emm(1)+spaceoffset,emm(2)-spaceoffset,'-/-');
scatter(1,1,'filled'); text(1+spaceoffset,1+spaceoffset,'Rejection')
title('Continuation Probabilities','FontWeight','normal')
xlim([-0.1 1.1]); xlabel('$\eta_1$: if approximation close','Interpreter','latex');
ylim([-0.1 1.1]); ylabel('$\eta_2$: if approximation far','Interpreter','latex');

%% Rare events
load('ABC_realisations.mat','sl');
for k=1:numel(sl)
    sl_mismatch_count(k,:) = [sum(sl{k}.kweights<0), sum(sl{k}.kweights>1)];
end
Fig3 = figure;
hist3(sl_mismatch_count,'ctrs',{0:5 0:5});
xlabel('False Positives');
ylabel('False Negatives');
zlabel('Frequency of realisations');
title('Realisations with observed false predictions','FontWeight','normal');
view([60 25]);

%% Fig 5: Efficiency across fixed number of proposals: 

% Important eta efficiencies
eff_sl = ESS_sl_ess ./ comptime_sl_ess; % Effiency of symmetric lazy (optimal)
eff_lz = ESS_lz_ess ./ comptime_lz_ess; % Efficiency of lazy (optimal)
eff_ml = ESS_ml_ess ./ comptime_ml_ess; % Efficiency of lazy (optimal)
eff_abc = ESS_abc_post(1) ./ comptime_abc_post(1); % The benchmark efficiency

% Fig5 = figure;
% set(gcf, 'Units', 'Inches', 'Position', [0, 0, 5, 6], 'PaperUnits', 'Inches', 'PaperSize', [5 6])
% 
% subplot(3,1,1), histogram(eff_sl,50,'Normalization','pdf');
% hold on, plot([eff_abc eff_abc],ylim,'k:','LineWidth',2); plot([mean(eff_sl) mean(eff_sl)],ylim,'r','LineWidth',2);
% xlim([0.005 0.075]); yticklabels([]); yticks([]);
% title('Multifidelity: $\eta_i = \hat{\eta}_i$','Interpreter','latex');
% 
% subplot(3,1,2), histogram(eff_ml,50,'Normalization','pdf'); 
% hold on, plot([eff_abc eff_abc],ylim,'k:','LineWidth',2); plot([mean(eff_ml) mean(eff_ml)],ylim,'r','LineWidth',2);
% xlim([0.005 0.075]); yticklabels([]); yticks([]);
% title('Multilevel: $\eta_1 = \eta_2$','Interpreter','latex');
% 
% subplot(3,1,3), histogram(eff_lz,50,'Normalization','pdf'); 
% hold on, plot([eff_abc eff_abc],ylim,'k:','LineWidth',2); plot([mean(eff_lz) mean(eff_lz)],ylim,'r','LineWidth',2);
% xlim([0.005 0.075]); yticklabels([]); yticks([]);
% xlabel('Effective samples per second')
% title('Early rejection: $\eta_1 = 1$','Interpreter','latex');

% Try rotating

Fig5 = figure;
set(gcf, 'Units', 'Inches', 'Position', [0, 0, 8, 3], 'PaperUnits', 'Inches', 'PaperSize', [8 3])

subplot(1,3,1), histogram(eff_sl,40,'Normalization','pdf');
ylim1 = ylim();
hold on, plot([eff_abc eff_abc],ylim1,'k:','LineWidth',2); plot([mean(eff_sl) mean(eff_sl)],ylim1,'r','LineWidth',2);
%set(gca,'view',[90 -90]);
axis square
xlim([0.005 0.075]); yticklabels([]); yticks([]);
ylim(ylim1);
xlabel('Effective samples per second','Interpreter','latex')
title('Early accept/reject: $(\eta_1,\eta_2) = (\hat{\eta}_1, \hat{\eta}_2)$','Interpreter','latex');

subplot(1,3,2), histogram(eff_ml,40,'Normalization','pdf'); 
ylim2 = ylim();
hold on, plot([eff_abc eff_abc],ylim2,'k:','LineWidth',2); plot([mean(eff_ml) mean(eff_ml)],ylim2,'r','LineWidth',2);
%set(gca,'view',[90 -90]);
axis square
xlim([0.005 0.075]); 
xlabel('Effective samples per second','Interpreter','latex')
yticklabels([]); yticks([]);
ylim(ylim2);
title('Early decision: $\eta_1 = \eta_2$','Interpreter','latex');

subplot(1,3,3), histogram(eff_lz,20,'Normalization','pdf'); 
ylim3 = ylim();
hold on, plot([eff_abc eff_abc],ylim3,'k:','LineWidth',2); plot([mean(eff_lz) mean(eff_lz)],ylim3,'r','LineWidth',2);
%set(gca,'view',[90 -90]);
axis square
xlabel('Effective samples per second','Interpreter','latex')
xlim([0.005 0.075]); yticklabels([]); yticks([]); 
ylim(ylim3);
title('Early rejection: $\eta_1 = 1$','Interpreter','latex');

% Empirical pairwise comparison of efficiencies
ESSF = [eff_sl' eff_ml' eff_lz' eff_abc*ones(size(eff_sl'))];

for a = 1:4
    for b = 1:4
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

%% Cloud of etas

[sl_cloud,lz_cloud,ml_cloud] = eta_cloud(benchmark,epsilon,0.1,250);

Fig6 = figure;
hold on
scatter(sl_cloud(:,1),sl_cloud(:,2),'filled'), hold on, scatter(eta_sl_ess(1),eta_sl_ess(2),'filled','MarkerFaceColor','red')
xlim([0 0.25]); ylim([0 0.20]);
xticks(0:0.05:0.25); xtickformat('%.2f');
yticks(0:0.05:0.20); ytickformat('%.2f');

mesh = 0:0.005:0.25;
phi = getPhiValues(mesh,benchmark,epsilon);
contour(mesh,mesh,phi,min(min(phi))./[.99 .95 .9 .75],'k--');

xlabel('$\eta_1$: continuation prob. if $\tilde y \in \tilde \Omega$','Interpreter','latex');
ylabel('$\eta_2$: continuation prob. if $\tilde y \notin \tilde \Omega$','Interpreter','latex');
box on
title('Burn-in estimates of $(\hat{\eta}_1,\hat{\eta}_2)$','Interpreter','latex');

%% Save all
FolderName = strcat('figs');   % Your destination folder

set(Fig1,'renderer','Painters');
saveas(Fig1,strcat(FolderName,'/Fig1'),'epsc');
saveas(Fig2a,strcat(FolderName,'/Fig2a'),'epsc');
saveas(Fig2b,strcat(FolderName,'/Fig2b'),'epsc');
saveas(Fig3,strcat(FolderName,'/Fig3'),'epsc');
saveas(Fig4,strcat(FolderName,'/Fig4'),'epsc');
set(Fig5,'renderer','Painters');
saveas(Fig5,strcat(FolderName,'/Fig5'),'epsc');
saveas(Fig6,strcat(FolderName,'/Fig6'),'epsc');

%% Functions

function [sl_cloud, lz_cloud, ml_cloud] = eta_cloud(benchmark,epsilon,percentage,replicates)

N = numel(benchmark.c);
M = floor(percentage*N);

for r = 1:replicates
    shuffle = randperm(N);
    idx = shuffle(1:M);
    sl_cloud(r,:) = optimal_sl_eta(benchmark.d(:,idx), epsilon, benchmark.c(idx), benchmark.delta(idx), benchmark.kproposals(:,idx), 0);
    lz_cloud(r,:) = optimal_lz_eta(benchmark.d(:,idx), epsilon, benchmark.c(idx), benchmark.delta(idx), benchmark.kproposals(:,idx), 0);
    ml_cloud(r,:) = optimal_ml_eta(benchmark.d(:,idx), epsilon, benchmark.c(idx), benchmark.delta(idx), benchmark.kproposals(:,idx), 0);
end

end

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