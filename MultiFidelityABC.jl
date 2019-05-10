module MultiFidelityABC
include("simmethods.jl")
include("mfabc.jl")
include("produce_figs.jl")
end
include("Viral.jl")
include("Repressilator.jl")
using .MultiFidelityABC
using LaTeXStrings

### PRODUCE HUGE SIMULATIONS (only needed once)
#= 
Best to begin Julia with parallel workers set up using:
% julia -p 10
for 10 parallel workers, for example.
=#
# BenchmarkCloud(Repressilator.mf_prob, 5000000, "./repressilator/output/")
# BenchmarkCloud(Viral.mf_prob, 5000000, "./repressilator/output/")

#=
## SET UP

# Import the output from the repressilator simulations
bm = BenchmarkCloud("./repressilator/output/")
sim_sets = Iterators.partition(bm, 10^4)
sim_set = collect(first(sim_sets))

epsilons = (50.0, 50.0)
F1(k) = (1.9<k[2]<2.1)
F2(k) = (1.5<k[2]<1.6)
F3(k) = k[2]
F = [F1, F2, F3]
budgets = [100.0,200.0,300.0]
=#

#=
## PRODUCE FIGURES

# Fig 1a
fig1a = view_distances(sim_set, epsilons)
# Fig 1b
fig1b = view_distances(sim_set, epsilons, 2, L"n")

# Fig 2a
fig2a = compare_efficiencies(bm, sim_sets, epsilons, output="theory")
# Fig 2b, Table 1
fig2b, t1, t1_latex = compare_efficiencies(bm, sim_sets, epsilons, output="plot")

# Table 2, 3 (and table 2 insert)
t2,t3,t2etas = observed_variances(bm, sim_sets, epsilons, F, budgets)

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