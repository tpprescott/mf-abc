function [dn] = getTauFolder(tau)
%getTauFolder: get the subfolder that would correspond to this value of tau

dn = sprintf('tau%.2e',tau);
end

