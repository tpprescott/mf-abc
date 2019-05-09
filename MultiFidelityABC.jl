module MultiFidelityABC
include("simmethods.jl")
include("mfabc.jl")
include("produce_figs.jl")
end
include("Viral.jl")
include("Repressilator.jl")
using .MultiFidelityABC
using LaTeXStrings

#
## SET UP
# Import the output from the repressilator simulations
allsims = BenchmarkCloud("./output/")
splitsims = Iterators.partition(allsims, 10^4)
bm = collect(first(splitsims))
simset = Iterators.drop(splitsims,1)

epsilons = (50.0, 50.0)
F1(k) = (1.9<k[2]<2.1)
F2(k) = (1.5<k[2]<1.6)
F3(k) = k[2]
F = [F1, F2, F3]
budgets = [100.0,200.0,300.0]
#=
## PRODUCE FIGURES

# Fig 1a
fig1a = view_distances(bm, epsilons)
# Fig 1b
fig1b = view_distances(bm, epsilons, 2, L"n")

# Fig 2a
fig2a = compare_efficiencies(bm, simset, epsilons, output="theory")
# Fig 2b
fig2b = compare_efficiencies(bm, simset, epsilons, output="plot")

# Table 1
t1 = compare_efficiencies(bm, simset, epsilons, output="table")

# Table 2, 3 (and table 2 insert)
t2,t3,t2etas = observed_variances(bm, simset, epsilons, F, budgets)

## SET UP
# Import the output from the viral dynamics simulations
allsims = BenchmarkCloud("./output/")
splitsims = Iterators.partition(allsims, 10^4)
bm = collect(first(splitsims))
simset = Iterators.drop(splitsims,1)

epsilons = (??)

## PRODUCE FIGURES
# Fig 3a
fig3a = view_distances(bm, epsilons)
# Fig 3b
fig3b = view_distances(bm, epsilons, 1, L"k_1")

# Fig 4
fig4a = efficiency_histogram(bm, simset, epsilons, method = "mf")
fig4b = efficiency_histogram(bm, simset, epsilons, method = "ed")
fig4c = efficiency_histogram(bm, simset, epsilons, method = "er")

=#