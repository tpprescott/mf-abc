function [vir_output, simtimes] = fullsimpop(T,x0,N,S,propfun)
% Daily increment to total virus output from each cell

vir_output = zeros(N,floor(T));
simtimes = zeros(N,1);

%parfor n=1:N
for n=1:N
    tic;
    [t,x] = gillespieDM_noPP(T,x0,S,propfun);
    simtimes(n) = toc;
    
    vo = interp1(t,x(4,:),0:1:T);
    vir_output(n,:) = diff(vo);
end

end
