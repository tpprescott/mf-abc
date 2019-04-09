function [vir_output, simtimes] = completesimpop(T,x0,PPpop,propfun,S,slowidx)
% Daily increment to total virus output from each cell

N = numel(PPpop);
vir_output = zeros(N,floor(T));
simtimes = zeros(N,1);

%parfor n=1:N
for n=1:N
    tic;
    [t,x] = gillespiePP_partition(T,x0,propfun,S,PPpop{n},slowidx);
    simtimes(n) = toc;
    
    vo = interp1(t,x(4,:),0:1:T);
    vir_output(n,:) = diff(vo);
end

end

