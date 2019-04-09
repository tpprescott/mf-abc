function fullPP = PPBridge_incexact(eCount,intLength,existingEvents)
% Function to return a full unit rate Poisson process, conditional on
% information about the number of events (eCount) in intervals of
% given lengths intLength.

intN = length(eCount);
intEnd = cumsum(intLength);
intEnd = [0 intEnd];
fullPP = zeros(1,sum(eCount));

event_i = 1;

for j = 1:intN
    fullPP_inc = intEnd(j) + intLength(j)*rand(1,eCount(j));
    fullPP(event_i:event_i+eCount(j)-1) = fullPP_inc;
    event_i = event_i + eCount(j);
end

fullPP = [fullPP existingEvents'];
fullPP = sort(fullPP);

end