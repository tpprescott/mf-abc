module MultiFidelityABC
include("simmethods.jl")
include("mfabc.jl")
include("produce_figs.jl")
end
include("Viral.jl")
include("Repressilator.jl")
using .MultiFidelityABC

allsims = BenchmarkCloud("./output/")
splitsims = Iterators.partition(allsims, 10^4)
bm = collect(first(splitsims))
simset = Iterators.drop(splitsims,1)