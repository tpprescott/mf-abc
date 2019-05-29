using Distributed
@everywhere include("MultiFidelityABC.jl")

bm = MakeBenchmarkCloud("viral/output")
idx = Iterators.partition(1:length(bm), 1000)

adaptive_mfabc(i) = MakeMFABCCloud(Viral.mf_prob, bm[i], (0.4,0.4), method="mf", lower_eta=0.1, budget=2*cost(bm[i]))
mfabc_set = pmap(adaptive_mfabc, idx)

for i in 1:length(mfabc_set)
write_cloud(mfabc_set[i], "viral/output/mf/cloud$i")
end
