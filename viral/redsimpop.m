function [vir_output, simtimes, PPpop] = redsimpop(T,x0,N,S,propfun,xcorrfun)
% Daily increment to total virus output from each cell - also the
% underlying random noise generation

vir_output = zeros(N,floor(T));
simtimes = zeros(N,1);
PPpop = cell(N,1);

%parfor n=1:N
for n=1:N
    tic;
    [t,x,PPpop{n}] = gillespie_hybrid(T,x0,S,propfun,xcorrfun);
    simtimes(n) = toc;
    
    vo = interp1(t,x(4,:),0:1:T);
    vir_output(n,:) = diff(vo);
end

end
