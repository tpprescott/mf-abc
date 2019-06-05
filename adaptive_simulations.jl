using Distributed
@everywhere include("MultiFidelityABC.jl")

bm = MakeBenchmarkCloud("viral/output")
idx = Iterators.partition(1:length(bm), 1000)
epsilons = (0.25, 0.25)
println("--- Loaded benchmark")

pmap(i->MakeMFABCCloud(Viral.mf_prob, bm[[i]], epsilons, method="mf", lower_eta=0.01, budget=1.01*cost(bm[[i]])), 1:nworkers());
println("--- Good to go")

println("--- Small burn-in")
mfabc_set = pmap(i->MakeMFABCCloud(Viral.mf_prob, bm[i], epsilons, method="mf", lower_eta=0.01, budget=2*cost(bm[i])), idx)
println("--- Completed extra simulations: small burn-in")
println("--- Writing files")
for i in 1:length(mfabc_set)
 write_cloud(mfabc_set[i], "viral/output/mf_smallBI/cloud$i")
end

println("--- Large burn-in")
mfabc_set = pmap(i->MakeMFABCCloud(Viral.mf_prob, bm, epsilons, method="mf", lower_eta=0.01, budget=cost(bm)+cost(bm[i])), idx)
println("--- Completed extra simulations: large burn-in")
println("--- Writing files")
for i in 1:length(mfabc_set)
 write_cloud(mfabc_set[i], "viral/output/mf_largeBI/cloud$i")
end


