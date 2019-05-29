using Distributed
@everywhere include("MultiFidelityABC.jl")

bm = MakeBenchmarkCloud("viral/output")
idx = Iterators.partition(1:length(bm), 1000)
println("--- Loaded benchmark")

pmap(i->MakeMFABCCloud(Viral.mf_prob, bm[[i]], (0.4,0.4), method="mf", lower_eta=0.1, budget=2*cost(bm[[i]])), 1:nworkers());
println("--- Good to go")

mfabc_set = pmap(i->MakeMFABCCloud(Viral.mf_prob, bm[i], (0.4,0.4), method="mf", lower_eta=0.1, budget=2*cost(bm[i])), idx)
println("--- Completed extra simulations")
println("--- Writing files")

for i in 1:length(mfabc_set)
write_cloud(mfabc_set[i], "viral/output/mf/cloud$i")
end
