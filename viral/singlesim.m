clear all
path('../simulation',path);

k = [1;
    0.025;
    100;
    0.25;
    1.9985;
    7.5e-5];

x0 = [1;
    0;
    0;
    0];

S = [0 1 0 0;
    1 -1 0 0;
    0 0 1 0;
    -1 0 0 0;
    0 0 -1 0;
    0 -1 -1 1];
S = S';

T = 200;

%% FULL SCALE SIMULATION
propfun = @(x)fullpropensity(x,k);

%tic;
%[t,x] = gillespieDM(T,x0,S,propfun);
%totcost = toc;
%return

%% REDUCED SIMULATION
redreactionIdx = [1 2 4 6];
redpropfun = @(x)redpropensity(x,k,redreactionIdx);
xcorrfun = @(x)xcorrection(x,k);
redS = S(:,redreactionIdx);

infFlag = 0;
while infFlag == 0
tic;
[tt,xx,redPP] = gillespie_hybrid(T,x0,redS,redpropfun,xcorrfun);
c = toc;

if xx(end,end)>20
    infFlag = 1; % Ensure we look at an infected cell.
end

end

%% CORRECT REDUCED SIMULATION
fastreactionIdx = [3 5];
fastS = S(:,fastreactionIdx);
fastpropfun = @(x)redpropensity(x,k,fastreactionIdx);

tic;
[tc,xc] = gillespiePP_partition(T,x0,propfun,S,redPP,redreactionIdx);
Delta = toc;

%% PLOT

% Full-scale simulation
%figure; semilogy(t,x); legend('template','genome','struct','virus')
%figure; plot(t,x); legend('template','genome','struct','virus'); ylim([0 10]);

figure; 
subplot(1,2,1), semilogy(tt,xx); legend('template','genome','struct','virus','Location','southeast'); xlim([0 T]); ylim([1 1e4]); 
title('Hybrid (low-fidelity)','FontWeight','normal');
xlabel('Days'); ylabel('Molecule count');
text(5,5000,sprintf('Simulation time\n%.2f s',c));
axis square

subplot(1,2,2), semilogy(tc,xc); 
%legend('template','genome','struct','virus','Location','southeast'); 
xlim([0 T]); ylim([1 1e4]); 
title('Gillespie (high-fidelity)','FontWeight','normal');
xlabel('Days'); %ylabel('Molecule count');
text(5,5000,sprintf('Additional simulation time\n%.2f s',Delta));
axis square


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